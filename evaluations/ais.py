"""
The model implements Hamiltonian AIS.
Code modified from Tensorflow implementation at https://github.com/jiamings/ais/
"""

import numpy as np
from evaluations.hmc import *
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm


# See if this should be in torch
def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))

class AISEnergyBasedModels(object):
    def __init__(self, model_energy_fn, dims, device,
                 stepsize=0.01, n_steps=10,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        Compute normalization constant of f(x) = exp(-model_energy_fn(x))

        Prior is a normal distribution with mean 0 and identity covariance matrix
        :param model_energy_fn: outputs e(x).
        :param dims e.g. {'output_dim': 28*28, 'input_dim': FLAGS.d, 'batch_size': 1} :)
        The following are parameters for HMC.
        :param stepsize:
        :param n_steps:
        :param target_acceptance_rate:
        :param avg_acceptance_slowness:
        :param stepsize_min:
        :param stepsize_max:
        :param stepsize_dec:
        :param stepsize_inc:
        """

        self.dims = dims
        self.model_energy_fn = model_energy_fn
        self.prior = MultivariateNormal(loc=torch.zeros(dims['input_dim']).to(device),
                                        covariance_matrix=torch.eye(dims['input_dim']).to(device))
        self.batch_size = dims['batch_size']
        self.stepsize = stepsize
        self.avg_acceptance_rate = target_acceptance_rate
        self.n_steps = n_steps
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_dec = stepsize_dec
        self.stepsize_inc = stepsize_inc
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_slowness = avg_acceptance_slowness

    def log_f_i(self, x, t):
        return -self.energy_fn(x, t)

    def energy_fn(self, x, t):
        return -(1 - t) * self.prior.log_prob(x) + t * self.model_energy_fn(x)

    def ais(self, schedule):
        """
            :param schedule: temperature schedule
        """
        x = self.prior.sample((self.batch_size,))
        w = torch.zeros_like(x[:, 0])
        stepsize = self.stepsize
        avg_acceptance_rate = self.avg_acceptance_rate

        for i, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
            new_u = self.log_f_i(x, t1)
            prev_u = self.log_f_i(x, t0)
            w += new_u - prev_u

            def run_energy(x):
                return self.energy_fn(x, t1)

            accept, final_pos, final_vel = hmc_move(
                x,
                run_energy,
                stepsize,
                self.n_steps
            )

            x, stepsize, avg_acceptance_rate = hmc_updates(
                x,
                stepsize,
                avg_acceptance_rate=avg_acceptance_rate,
                final_pos=final_pos,
                accept=accept,
                stepsize_min=self.stepsize_min,
                stepsize_max=self.stepsize_max,
                stepsize_dec=self.stepsize_dec,
                stepsize_inc=self.stepsize_inc,
                target_acceptance_rate=self.target_acceptance_rate,
                avg_acceptance_slowness=self.avg_acceptance_slowness
            )

        # Return w and the log_mean_exp:
        return (w, torch.logsumexp(w, dim=0) - np.log(w.size(0)))


class AISLatentVariableModels(object):
    def __init__(self, recon_energy, dims, device,
                 stepsize=0.01, n_Ts=1000, n_steps=10, n_chains=16,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        Compute the likelihood for a latent variable model with Gaussian prior

        Prior is a normal distribution with mean 0 and identity covariance matrix
        :param model_energy_fn: outputs e(x).
        :param dims e.g. {'output_dim': 28*28, 'input_dim': FLAGS.d, 'batch_size': 1} :)
        The following are parameters for HMC.
        :param stepsize:
        :param n_steps:
        :param target_acceptance_rate:
        :param avg_acceptance_slowness:
        :param stepsize_min:
        :param stepsize_max:
        :param stepsize_dec:
        :param stepsize_inc:
        """

        self.dims = dims
        self.device = device
        self.recon_energy = recon_energy
        self.prior = MultivariateNormal(loc=torch.zeros(dims).to(device),
                                        covariance_matrix=torch.eye(dims).to(device))
        self.stepsize = stepsize
        self.avg_acceptance_rate = target_acceptance_rate
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_dec = stepsize_dec
        self.stepsize_inc = stepsize_inc
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_slowness = avg_acceptance_slowness
        self.schedule = get_schedule(n_Ts, 4)

    def log_f_i(self, x, z, t):
        return -self.energy_fn(x, z, t)

    def energy_fn(self, x, z, t):
        return -self.prior.log_prob(z) + t * self.recon_energy(x, z)

    def ais(self, x):
        """
            :param schedule: temperature schedule
        """
        z = self.prior.sample((self.n_chains * x.shape[0],))
        w = torch.zeros(self.n_chains * x.shape[0], device=self.device)
        stepsize = self.stepsize
        avg_acceptance_rate = self.avg_acceptance_rate

        x = x.unsqueeze(0).expand(self.n_chains, -1, -1, -1, -1).contiguous().view(-1, x.shape[1], x.shape[2], x.shape[3])
        for i, (t0, t1) in tqdm(enumerate(zip(self.schedule[:-1], self.schedule[1:]))):
            with torch.no_grad():
                new_u = self.log_f_i(x, z, t1)
                prev_u = self.log_f_i(x, z, t0)
                w += new_u - prev_u

            def run_energy(z):
                return self.energy_fn(x, z, t1)

            accept, final_pos, final_vel = hmc_move(
                z,
                run_energy,
                stepsize,
                self.n_steps
            )

            z, stepsize, avg_acceptance_rate = hmc_updates(
                z,
                stepsize,
                avg_acceptance_rate=avg_acceptance_rate,
                final_pos=final_pos,
                accept=accept,
                stepsize_min=self.stepsize_min,
                stepsize_max=self.stepsize_max,
                stepsize_dec=self.stepsize_dec,
                stepsize_inc=self.stepsize_inc,
                target_acceptance_rate=self.target_acceptance_rate,
                avg_acceptance_slowness=self.avg_acceptance_slowness
            )


        w = w.view(self.n_chains, -1)
        w = torch.logsumexp(w, dim=0) - np.log(w.shape[0])
        return w
