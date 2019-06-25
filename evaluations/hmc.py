import torch
from tqdm import tqdm


def kinetic_energy(v):
    return 0.5 * (v ** 2).sum(1)


def hamiltonian(p, v, energy_fn):
    return energy_fn(p) + kinetic_energy(v)


def mh_accept(e_prev, e_next):
    e_diff = e_prev - e_next
    return (torch.exp(e_diff) - torch.rand_like(e_diff)) >= 0


def simulate_dynamics(pos, vel, step, n_steps, energy_fn):
    for i in range(n_steps):
        pos = pos.detach()
        vel = vel.detach()
        pos.requires_grad = True
        dE_dpos = torch.autograd.grad(energy_fn(pos).sum(), pos, create_graph=False, only_inputs=True)[0]
        with torch.no_grad():
            if i == 0 or i == n_steps - 1:
                vel = vel - 0.5 * step * dE_dpos
            else:
                vel = vel - step * dE_dpos
            if i != n_steps - 1:
                pos = pos + step * vel
                # print((vel.abs() * step < 0.0001).float().mean())
    return pos.detach(), vel.detach()


def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
    initial_vel = torch.randn_like(initial_pos)
    final_pos, final_vel = simulate_dynamics(
        pos=initial_pos,
        vel=initial_vel,
        step=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    accept = mh_accept(
        e_prev=hamiltonian(initial_pos, initial_vel, energy_fn),
        e_next=hamiltonian(final_pos, final_vel, energy_fn)
    )
    return accept, final_pos, final_vel


def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness):
    new_pos = torch.where(accept.view(-1, 1), final_pos, initial_pos)
    # print(((new_pos - final_pos).abs().mean(-1) < 0.01).float().mean())
    if avg_acceptance_rate > target_acceptance_rate:
        new_stepsize_ = stepsize * stepsize_inc
    else:
        new_stepsize_ = stepsize * stepsize_dec
    new_stepsize = max(min(new_stepsize_, stepsize_max), stepsize_min)
    new_acceptance_rate = avg_acceptance_slowness * avg_acceptance_rate + (
                1.0 - avg_acceptance_slowness) * accept.float().mean()
    return new_pos, new_stepsize, new_acceptance_rate


# NOTE: target acceptance rate default was previously 0.65
# Stepsize changed to 0.1 default, stepsize max changed to 1.
class HMCSampler(object):
    def __init__(self, model_energy_fn,
                 stepsize=0.01, n_steps=10,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=1.0, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        Compute samples from a given energy based model.

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

        self.model_energy_fn = model_energy_fn
        self.stepsize = stepsize
        self.avg_acceptance_rate = target_acceptance_rate
        self.n_steps = n_steps
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_dec = stepsize_dec
        self.stepsize_inc = stepsize_inc
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_slowness = avg_acceptance_slowness

    def run_hmc_sampler(self, x, num_steps):
        stepsize = self.stepsize
        avg_acceptance_rate = self.avg_acceptance_rate

        for _ in tqdm(range(num_steps)):
            accept, final_pos, final_vel = hmc_move(
                x,
                self.model_energy_fn,
                self.stepsize,
                self.n_steps
            )

            x, self.stepsize, avg_acceptance_rate = hmc_updates(
                x,
                self.stepsize,
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


        self.avg_acceptance_rate = avg_acceptance_rate
        return x