import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from models.nice_approxbp import Logit


class ParzenGaussian(nn.Module):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data
        self.log_sigma = nn.Parameter(torch.tensor(-0.3))
        self.dim = self.train_data.shape[1]

    def forward(self, inputs):
        XY = self.train_data @ inputs.t()
        XX = torch.norm(self.train_data, dim=-1) ** 2
        YY = torch.norm(inputs, dim=-1) ** 2
        dist = XX[:, None] - 2 * XY + YY[None, :]
        log_mix_p = -0.5 * torch.exp(-2 * self.log_sigma) * dist
        log_mix_p -= self.dim * (np.log(2. * np.pi) + self.log_sigma)
        logp = torch.logsumexp(log_mix_p, dim=0).mean()

        return logp


def select_sigma(train_data_iter, test_data_iter, noise_sigma=None, logit_mnist=True):
    train_data = []
    test_data = []
    try:
        while True:
            if logit_mnist:
                X, y = next(train_data_iter)
                X.clamp_(1e-3, 1 - 1e-3)
                flattened_X, _ = Logit()(X.view(X.shape[0], -1), mode='direct')
            else:
                flattened_X = next(train_data_iter)
            if noise_sigma is not None:
                flattened_X += torch.randn_like(flattened_X) * noise_sigma

            train_data.append(flattened_X)
    except:
        pass

    try:
        while True:
            if logit_mnist:
                X, y = next(test_data_iter)
                X.clamp_(1e-3, 1 - 1e-3)
                flattened_X, _ = Logit()(X.view(X.shape[0], -1), mode='direct')
            else:
                flattened_X = next(test_data_iter)
            if noise_sigma is not None:
                flattened_X += torch.randn_like(flattened_X) * noise_sigma

            test_data.append(flattened_X)
    except:
        pass

    train_data = torch.cat(train_data, dim=0)
    test_data = torch.cat(test_data, dim=0)

    model = ParzenGaussian(train_data)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(1000):
        test_idxes = np.random.choice(test_data.shape[0], 100)
        test_X = test_data[test_idxes]
        logp = model(test_X)
        loss = -logp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step: {}, loss: {}, sigma: {}".format(
            i, loss.item(), model.log_sigma.exp().item()
        ))

    return model.log_sigma.exp().item()


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss