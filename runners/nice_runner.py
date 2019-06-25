import os
import logging
import shutil
import tensorboardX
from losses.sliced_sm import *
from losses.dsm import dsm, select_sigma
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.nice_approxbp import Logit, approx_backprop_score_matching, NICE
from losses.score_matching import exact_score_matching
import numpy as np
import pickle
import copy

__all__ = ['NICERunner']


class NICERunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def evaluate_model(self, state_dict, model_type, val_loader, test_loader, model_path):
        torch.save(state_dict, os.path.join(model_path, model_type + "_nice.pth"))

        flow = NICE(self.config.input_dim, self.config.model.hidden_size, self.config.model.num_layers).to(
            self.config.device)
        flow.load_state_dict(state_dict)
        def energy_net(inputs):
            energy, _ = flow(inputs, inv=False)
            return -energy
        logging.info("Evaluating for " + model_type)
        logging.info("Evaluating on validation set!")
        val_logps = []
        val_sm_losses = []
        noise_sigma = self.config.data.noise_sigma
        self.results[model_type] = {}

        for i, (X, y) in enumerate(val_loader):
            X = X + (torch.rand_like(X) - 0.5) / 256.
            flattened_X = X.type(torch.float32).to(self.config.device).view(X.shape[0], -1)
            flattened_X.clamp_(1e-3, 1-1e-3)
            flattened_X, _ = Logit()(flattened_X, mode='direct')
            if noise_sigma is not None:
                flattened_X += torch.randn_like(flattened_X) * noise_sigma

            logp = -energy_net(flattened_X)
            logp = logp.mean()
            val_logps.append(logp)
            sm_loss = exact_score_matching(energy_net, flattened_X, train=False).mean()
            val_sm_losses.append(sm_loss)

        val_logp = sum(val_logps) / len(val_logps)
        val_sm_loss = sum(val_sm_losses) / len(val_sm_losses)
        self.results[model_type]['val_logp'] = np.asscalar(val_logp.detach().cpu().numpy())
        self.results[model_type]['val_sm_loss'] = np.asscalar(val_sm_loss.detach().cpu().numpy())
        logging.info("Val logp: {}, score matching loss: {}".format(val_logp.item(), val_sm_loss.item()))

        logging.info("Evaluating on test set!")
        test_logps = []
        test_sm_losses = []

        for i, (X, y) in enumerate(test_loader):
            X = X + (torch.rand_like(X) - 0.5) / 256.
            flattened_X = X.type(torch.float32).to(self.config.device).view(X.shape[0], -1)
            flattened_X.clamp_(1e-3, 1-1e-3)
            flattened_X, _ = Logit()(flattened_X, mode='direct')
            if noise_sigma is not None:
                flattened_X += torch.randn_like(flattened_X) * noise_sigma

            logp = -energy_net(flattened_X)
            logp = logp.mean()
            test_logps.append(logp)
            sm_loss = exact_score_matching(energy_net, flattened_X, train=False).mean()
            test_sm_losses.append(sm_loss)

        test_logp = sum(test_logps) / len(test_logps)
        test_sm_loss = sum(test_sm_losses) / len(test_sm_losses)
        self.results[model_type]['test_logp'] = np.asscalar(test_logp.detach().cpu().numpy())
        self.results[model_type]['test_sm_loss'] = np.asscalar(test_sm_loss.detach().cpu().numpy())

    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, val_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
            val_dataset = Subset(dataset, val_indices)
            dataset = Subset(dataset, train_indices)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True,
                                 transform=transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=2)

        val_iter = iter(val_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        model_path = os.path.join(self.args.run, 'results', self.args.doc)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        flow = NICE(self.config.input_dim, self.config.model.hidden_size, self.config.model.num_layers).to(
            self.config.device)

        optimizer = self.get_optimizer(flow.parameters())

        # Set up test data
        noise_sigma = self.config.data.noise_sigma
        step = 0

        def energy_net(inputs):
            energy, _ = flow(inputs, inv=False)
            return -energy

        def grad_net_kingma(inputs):
            energy, _ = flow(inputs, inv=False)
            grad1, grad2 = flow.grads_backward(inv=False)
            return -grad1, -grad2

        def grad_net_UT(inputs):
            energy, _ = flow(inputs, inv=False)
            grad1, T, U = flow.grads_backward_TU(inv=False)
            grad2 = T * U / 2.
            return -grad1, -grad2

        def grad_net_S(inputs):
            energy, _ = flow(inputs, inv=False)
            grad1, S_r, S_i = flow.grads_backward_S(inv=False)
            grad2 = (S_r ** 2 - S_i ** 2)
            return -grad1, -grad2

        def sample_net(z):
            samples, _ = flow(z, inv=True)
            samples, _ = Logit()(samples, mode='inverse')
            return samples

        # Use this to select the sigma for DSM losses
        if self.config.training.algo == 'dsm':
            sigma = self.args.dsm_sigma
            # if noise_sigma is None:
            #     sigma = select_sigma(iter(dataloader), iter(val_loader))
            # else:
            #     sigma = select_sigma(iter(dataloader), iter(val_loader), noise_sigma=noise_sigma)

        if self.args.load_path != "":
            flow.load_state_dict(torch.load(self.args.load_path))

        best_model = {"val": None, "ll": None, "esm": None}
        best_val_loss = {"val": 1e+10, "ll": -1e+10, "esm": 1e+10}
        best_val_iter = {"val": 0, "ll": 0, "esm": 0}

        for _ in range(self.config.training.n_epochs):
            for _, (X, y) in enumerate(dataloader):
                X = X + (torch.rand_like(X) - 0.5) / 256.
                flattened_X = X.type(torch.float32).to(self.config.device).view(X.shape[0], -1)
                flattened_X.clamp_(1e-3, 1-1e-3)
                flattened_X, _ = Logit()(flattened_X, mode='direct')

                if noise_sigma is not None:
                    flattened_X += torch.randn_like(flattened_X) * noise_sigma

                flattened_X.requires_grad_(True)

                logp = -energy_net(flattened_X)

                logp = logp.mean()

                if self.config.training.algo == 'kingma':
                    loss = approx_backprop_score_matching(grad_net_kingma, flattened_X)
                if self.config.training.algo == 'UT':
                    loss = approx_backprop_score_matching(grad_net_UT, flattened_X)
                if self.config.training.algo == 'S':
                    loss = approx_backprop_score_matching(grad_net_S, flattened_X)
                elif self.config.training.algo == 'mle':
                    loss = -logp
                elif self.config.training.algo == 'ssm':
                    loss, *_ = single_sliced_score_matching(energy_net, flattened_X, noise_type=self.config.training.noise_type)
                elif self.config.training.algo == 'ssm_vr':
                    loss, *_ = sliced_VR_score_matching(energy_net, flattened_X, noise_type=self.config.training.noise_type)
                elif self.config.training.algo == 'dsm':
                    loss = dsm(energy_net, flattened_X, sigma=sigma)
                elif self.config.training.algo == "exact":
                    loss = exact_score_matching(energy_net, flattened_X, train=True).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if step % 10 == 0:
                    try:
                        val_X, _ = next(val_iter)
                    except:
                        val_iter = iter(val_loader)
                        val_X, _ = next(val_iter)

                    val_X = val_X + (torch.rand_like(val_X) - 0.5) / 256.
                    val_X = val_X.type(torch.float32).to(self.config.device)
                    val_X.clamp_(1e-3, 1-1e-3)
                    val_X, _ = Logit()(val_X, mode='direct')
                    val_X = val_X.view(val_X.shape[0], -1)
                    if noise_sigma is not None:
                        val_X += torch.randn_like(val_X) * noise_sigma

                    val_logp = -energy_net(val_X).mean()
                    if self.config.training.algo == 'kingma':
                        val_loss = approx_backprop_score_matching(grad_net_kingma, val_X)
                    if self.config.training.algo == 'UT':
                        val_loss = approx_backprop_score_matching(grad_net_UT, val_X)
                    if self.config.training.algo == 'S':
                        val_loss = approx_backprop_score_matching(grad_net_S, val_X)
                    elif self.config.training.algo == 'ssm':
                        val_loss, *_ = single_sliced_score_matching(energy_net, val_X, noise_type=self.config.training.noise_type)
                    elif self.config.training.algo == 'ssm_vr':
                        val_loss, *_ = sliced_VR_score_matching(energy_net, val_X, noise_type=self.config.training.noise_type)
                    elif self.config.training.algo == 'dsm':
                        val_loss = dsm(energy_net, val_X, sigma=sigma)
                    elif self.config.training.algo == 'mle':
                        val_loss = -val_logp
                    elif self.config.training.algo == "exact":
                        val_loss = exact_score_matching(energy_net, val_X, train=False).mean()

                    logging.info("logp: {:.3f}, val_logp: {:.3f}, loss: {:.3f}, val_loss: {:.3f}".format(logp.item(),
                                                                                           val_logp.item(),
                                                                                           loss.item(),
                                                                                           val_loss.item()))
                    tb_logger.add_scalar('logp', logp, global_step=step)
                    tb_logger.add_scalar('loss', loss, global_step=step)
                    tb_logger.add_scalar('val_logp', val_logp, global_step=step)
                    tb_logger.add_scalar('val_loss', val_loss, global_step=step)

                    if val_loss < best_val_loss['val']:
                        best_val_loss['val'] = val_loss
                        best_val_iter['val'] = step
                        best_model['val'] = copy.deepcopy(flow.state_dict())
                    if val_logp > best_val_loss['ll']:
                        best_val_loss['ll'] = val_logp
                        best_val_iter['ll'] = step
                        best_model['ll'] = copy.deepcopy(flow.state_dict())

                if step % 100 == 0:
                    with torch.no_grad():
                        z = torch.normal(torch.zeros(100, flattened_X.shape[1], device=self.config.device))
                        samples = sample_net(z)
                        samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                               self.config.data.image_size)
                        samples = torch.clamp(samples, 0.0, 1.0)
                        image_grid = make_grid(samples, 10)
                        tb_logger.add_image('samples', image_grid, global_step=step)
                        data = X
                        data_grid = make_grid(data[:100], 10)
                        tb_logger.add_image('data', data_grid, global_step=step)

                    logging.info("Computing exact score matching....")
                    try:
                        val_X, _ = next(val_iter)
                    except:
                        val_iter = iter(val_loader)
                        val_X, _ = next(val_iter)

                    val_X = val_X + (torch.rand_like(val_X) - 0.5) / 256.
                    val_X = val_X.type(torch.float32).to(self.config.device)
                    val_X.clamp_(1e-3, 1-1e-3)
                    val_X, _ = Logit()(val_X, mode='direct')
                    val_X = val_X.view(val_X.shape[0], -1)
                    if noise_sigma is not None:
                        val_X += torch.randn_like(val_X) * noise_sigma

                    sm_loss = exact_score_matching(energy_net, val_X, train=False).mean()
                    if sm_loss < best_val_loss['esm']:
                        best_val_loss['esm'] = sm_loss
                        best_val_iter['esm'] = step
                        best_model['esm'] = copy.deepcopy(flow.state_dict())

                    logging.info('step: {}, exact score matching loss: {}'.format(step, sm_loss.item()))
                    tb_logger.add_scalar('exact_score_matching_loss', sm_loss, global_step=step)

                if step % 500 == 0:
                    torch.save(flow.state_dict(), os.path.join(model_path, 'nice.pth'))

                step += 1

        self.results = {}
        self.evaluate_model(flow.state_dict(), "final", val_loader, test_loader, model_path)
        self.evaluate_model(best_model['val'], "best_on_val", val_loader, test_loader, model_path)
        self.evaluate_model(best_model['ll'], "best_on_ll", val_loader, test_loader, model_path)
        self.evaluate_model(best_model['esm'], "best_on_esm", val_loader, test_loader, model_path)
        self.results['final']['num_iters'] = step
        self.results['best_on_val']['num_iters'] = best_val_iter['val']
        self.results['best_on_ll']['num_iters'] = best_val_iter['ll']
        self.results['best_on_esm']['num_iters'] = best_val_iter['esm']

        pickle_out = open(model_path + "/results.pkl", "wb")
        pickle.dump(self.results, pickle_out)
        pickle_out.close()