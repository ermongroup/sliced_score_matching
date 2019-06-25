import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.autograd as autograd
import logging
from torch.utils.data import DataLoader
import time
import models.nice_approxbp
from tqdm import tqdm

class DKEF(nn.Module):
    def __init__(self, input_dim, mode, num_kernels=1, init_z=None, M=None,
                 sigma_list=(0.0, 0.5185, 1.0), hidden_dim=30, add_skip=False,
                 alpha_param=False, train_Z=True, pretrained_encoder=None, dsm_sigma=None):
        super().__init__()
        self.num_kernels = num_kernels
        if mode not in ["exact", "sliced", "sliced_VR", "dsm", "kingma", "CP"]:
            raise ValueError("Unallowable training mode.")
        assert ((mode == "dsm") == (dsm_sigma is not None))
        self.mode = mode
        self.dsm_sigma = dsm_sigma
        self.input_dim = input_dim
        if self.mode == "kingma" or self.mode == "CP":
            self.kernel = nn.ModuleList([Kernel_approx_bp(self.input_dim,
                                                hidden_dim=hidden_dim, add_skip=add_skip, sigma_init=sigma_list[i],
                                                pretrained_encoder=pretrained_encoder) for i in range(self.num_kernels)])
        else:
            self.kernel = nn.ModuleList([Kernel(self.input_dim,
                                                hidden_dim=hidden_dim, add_skip=add_skip, sigma_init=sigma_list[i],
                                                pretrained_encoder=pretrained_encoder) for i in range(self.num_kernels)])
        self.kernel_weights = nn.Parameter(torch.zeros(num_kernels))
        self.q0_sigma = 2.0
        if init_z is not None:
            self.z = nn.Parameter(init_z, requires_grad=train_Z)
            self.M = init_z.shape[0]
        else:
            logging.info("Be careful about random initialization of z (generally not used)")
            self.M = M
            self.z = nn.Parameter(torch.randn((M, self.input_dim)), requires_grad=train_Z)

        # TODO: remove allowing alpha to be a parameter?
        self.alpha_param = alpha_param
        if alpha_param:
            self.alpha = nn.Parameter(torch.ones(self.M) / self.M)
        else:
            self.alpha = torch.zeros(self.M)
        self.log_lambd = nn.Parameter(torch.tensor(-2.0))

    def update_alpha(self, data, stage="train"):
        if stage == "finalize":
            G = self.G / self.nsamples
            b = self.b / self.nsamples
        elif stage == "train":
            if "sliced" in self.mode:
                vectors = torch.randn_like(data).sign()
            if self.mode == "dsm":
                vectors = torch.randn_like(data) * self.dsm_sigma
                data = data + vectors

            data.requires_grad_(True)
            q0_grad = autograd.grad(self.q0(data).sum(), data)[0]
            kernel_weight = torch.softmax(self.kernel_weights, dim=0)

            if self.mode == "kingma":
                kxz = sum([kernel_weight[j] * self.kernel[j].forward(data, self.z) for j in range(self.num_kernels)])
                grads = [self.kernel[j].grads_backward() for j in range(self.num_kernels)]
                grad1s = [grad[0] for grad in grads]
                grad2s = [grad[1] for grad in grads]
                kxz_gradx = sum([kernel_weight[j] * grad1s[j] for j in range(self.num_kernels)])
                kxz_gradx2 = sum([kernel_weight[j] * grad2s[j] for j in range(self.num_kernels)])
                G = (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).mean(0)
                b = kxz_gradx2.sum(2).mean(0)
            elif self.mode == "CP":
                kxz = sum([kernel_weight[j] * self.kernel[j].forward(data, self.z) for j in range(self.num_kernels)])
                grads = [self.kernel[j].grads_backward_S(
                    grad1=torch.ones_like(kxz) * kernel_weight[j], S_r=torch.zeros_like(kxz), S_i=torch.zeros_like(kxz))
                    for j in range(self.num_kernels)]
                kxz_gradx = sum([grad[0] for grad in grads])
                kxzS_r = sum([grad[1] for grad in grads])
                kxzS_i = sum([grad[2] for grad in grads])

                kxz_gradx2 = kxzS_r ** 2 - kxzS_i ** 2
                G = (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).mean(0)
                b = kxz_gradx2.sum(2).mean(0)
            else:
                data = data.unsqueeze(1).expand(-1, self.M, -1)
                data.requires_grad_(True)
                kxz = sum([kernel_weight[j] * self.kernel[j].alpha_forward(data, self.z) for j in range(self.num_kernels)])
                kxz_gradx = autograd.grad(kxz.sum(), data, create_graph=True)[0]

                if self.mode == "exact":
                    b1 = torch.zeros(self.M, device=data.device)
                    for j in range(data.shape[2]):
                        grad = autograd.grad(kxz_gradx[:, :, j].sum(), data, create_graph=True,
                                             retain_graph=True)[0][:, :, j].mean(0)
                        b1 += grad

                    G = (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).mean(0)
                    b2 = (kxz_gradx @ q0_grad[:, :, None]).squeeze().mean(0)
                    b = b1 + b2
                elif self.mode == "sliced":
                    q0_gradv = (q0_grad * vectors).sum(1)
                    kxz_gradxv = (kxz_gradx * vectors[:,None,:]).sum(2)
                    G = (kxz_gradxv[:,None,:] * kxz_gradxv[:,:,None]).mean(0)
                    grad2 = autograd.grad(kxz_gradxv.sum(), data, create_graph=True)[0]
                    b1 = (grad2 * vectors[:,None,:]).sum(2).mean(0)
                    b2 = (kxz_gradxv * q0_gradv[:,None]).mean(0)
                    b = b1 + b2
                elif self.mode == "sliced_VR":
                    G = (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).mean(0)
                    kxz_gradxv = (kxz_gradx * vectors[:,None,:]).sum(2)
                    grad2 = autograd.grad(kxz_gradxv.sum(), data, create_graph=True)[0]
                    b1 = (grad2 * vectors[:,None,:]).sum(2).mean(0)
                    b2 = (kxz_gradx @ q0_grad[:, :, None]).squeeze().mean(0)
                    b = b1 + b2
                elif self.mode == "dsm":
                    G = (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).mean(0) * self.dsm_sigma ** 4
                    prod_term = self.dsm_sigma ** 2 * q0_grad[:, :, None] + vectors[:,:,None]
                    b = self.dsm_sigma ** 2 * (kxz_gradx @ prod_term).squeeze().mean(0)

        matrix = G + torch.pow(10., self.log_lambd) * torch.eye(self.M, device=G.device)
        soln, _ = torch.gesv(-b[:,None], matrix)
        soln = soln[:,0]

        self.alpha.data = soln
        return soln

    def zero_alpha_matrices(self, device):
        self.G = torch.zeros((self.M, self.M), device=device)
        self.b = torch.zeros((self.M,), device=device)
        self.nsamples = 0.

    def accumulate_alpha_matrices(self, data, vectors=None):
        self.nsamples += data.size(0)
        data.requires_grad_(True)
        q0_grad = autograd.grad(self.q0(data).sum(), data)[0]
        kernel_weight = torch.softmax(self.kernel_weights, dim=0)

        if self.mode == "kingma":
            kxz = sum([kernel_weight[j] * self.kernel[j].forward(data, self.z) for j in range(self.num_kernels)])
            grads = [self.kernel[j].grads_backward() for j in range(self.num_kernels)]
            grad1s = [grad[0] for grad in grads]
            grad2s = [grad[1] for grad in grads]
            kxz_gradx = sum([kernel_weight[j] * grad1s[j] for j in range(self.num_kernels)])
            kxz_gradx2 = sum([kernel_weight[j] * grad2s[j] for j in range(self.num_kernels)])
            self.G += (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).sum(0).detach()
            self.b += kxz_gradx2.sum(2).sum(0).detach()
        elif self.mode == "CP":
            kxz = sum([kernel_weight[j] * self.kernel[j].forward(data, self.z) for j in range(self.num_kernels)])
            grads = [self.kernel[j].grads_backward_S(
                grad1=torch.ones_like(kxz) * kernel_weight[j], S_r=torch.zeros_like(kxz), S_i=torch.zeros_like(kxz))
                for j in range(self.num_kernels)]
            kxz_gradx = sum([grad[0] for grad in grads])
            kxzS_r = sum([grad[1] for grad in grads])
            kxzS_i = sum([grad[2] for grad in grads])
            kxz_gradx2 = kxzS_r ** 2 - kxzS_i ** 2
            self.G += (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).sum(0).detach()
            self.b += kxz_gradx2.sum(2).sum(0).detach()
        else:
            data = data.unsqueeze(1).repeat(1, self.M, 1)
            data.requires_grad_(True)
            kxz = sum([kernel_weight[j] * self.kernel[j].alpha_forward(data, self.z) for j in range(self.num_kernels)])
            kxz_gradx = autograd.grad(kxz.sum(), data, create_graph=True)[0]

            if self.mode == "exact":
                b1 = torch.zeros(self.M, device=data.device)
                for j in range(data.shape[2]):
                    grad = autograd.grad(kxz_gradx[:, :, j].sum(), data, retain_graph=True)[0][:, :, j].sum(0)
                    b1 += grad.detach()

                self.G += (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).sum(0).detach()
                b2 = (kxz_gradx @ q0_grad[:, :, None]).squeeze().sum(0)
                self.b += (b1 + b2).detach()
            elif self.mode == "sliced":
                q0_gradv = (q0_grad * vectors).sum(1)
                kxz_gradxv = (kxz_gradx * vectors[:, None, :]).sum(2)
                self.G += (kxz_gradxv[:,None,:] * kxz_gradxv[:,:,None]).sum(0).detach()
                grad2 = autograd.grad(kxz_gradxv.sum(), data, create_graph=True)[0]
                b1 = (grad2 * vectors[:, None, :]).sum(2).sum(0)
                b2 = (kxz_gradxv * q0_gradv[:, None]).sum(0)
                self.b += (b1 + b2).detach()
            elif self.mode == "sliced_VR":
                self.G += (kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).sum(0).detach()
                kxz_gradxv = (kxz_gradx * vectors[:,None,:]).sum(2)
                grad2 = autograd.grad(kxz_gradxv.sum(), data, create_graph=True)[0]
                b1 = (grad2 * vectors[:,None,:]).sum(2).sum(0)
                b2 = (kxz_gradx @ q0_grad[:, :, None]).squeeze().sum(0)
                self.b += (b1 + b2).detach()
            elif self.mode == "dsm":
                self.G += ((kxz_gradx @ torch.transpose(kxz_gradx, 1, 2)).sum(0) * self.dsm_sigma ** 4).detach()
                prod_term = self.dsm_sigma ** 2 * q0_grad[:, :, None] + vectors[:, :, None]
                self.b += (self.dsm_sigma ** 2 * (kxz_gradx @ prod_term).squeeze().sum(0)).detach()


    def save_alpha_matrices(self, data, collate_fn, device, batch_size=100, override=False):
        if self.alpha_param and not override:
            return

        self.zero_alpha_matrices(device=device)

        loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
        for data in loader:
            data = data.to(device)
            if "sliced" in self.mode:
                vectors = torch.randn_like(data).sign()
            elif self.mode == "dsm":
                vectors = torch.randn_like(data) * self.dsm_sigma
                data = data + vectors
            else:
                vectors = None

            self.accumulate_alpha_matrices(data, vectors=vectors)

        self.update_alpha(None, stage="finalize")

    def q0(self, x):
        q0 = - 0.5 / (self.q0_sigma ** 2) * torch.sum(x ** 2, dim=1)
        return q0

    def q0_kingma(self, x):
        q0 = - 0.5 / (self.q0_sigma ** 2) * torch.sum(x ** 2, dim=1)
        grad1 = - x / self.q0_sigma ** 2
        grad2 = - torch.ones_like(x) / self.q0_sigma ** 2
        return q0, grad1, grad2

    def q0_S(self, x):
        q0 = - 0.5 / (self.q0_sigma ** 2) * torch.sum(x ** 2, dim=1)
        grad1 = - x / self.q0_sigma ** 2
        S_r = torch.zeros_like(x)
        v = torch.randn_like(S_r).sign()
        S_i = v / self.q0_sigma
        return q0, grad1, S_r, S_i

    # Options for stage: train, finalize, eval
    def forward(self, x_t, x_v, stage="train"):
        if self.alpha_param or stage == "eval":
            alpha = self.alpha
        else:
            alpha = self.update_alpha(x_t, stage=stage)
        kernel_weight = torch.softmax(self.kernel_weights, dim=0)

        log_px = self.q0(x_v)
        for (kernel, kw) in zip(self.kernel, kernel_weight):
            kxz = kernel(x_v, self.z)
            log_px += kxz @ alpha * kw

        return log_px

    def approx_bp_forward(self, x_t, x_v, stage="train", mode="kingma"):
        if self.alpha_param or stage == "eval":
            alpha = self.alpha
        else:
            alpha = self.update_alpha(x_t, stage=stage)
        kernel_weight = torch.softmax(self.kernel_weights, dim=0)

        if mode == "kingma":
            log_px, grad1, grad2 = self.q0_kingma(x_v)
            for (kernel, kw) in zip(self.kernel, kernel_weight):
                kxz = kernel(x_v, self.z)
                log_px += kxz @ alpha * kw
                kxz_gradx, kxz_gradx2 = kernel.grads_backward(alpha[None, :, None] * kw,
                                                              torch.zeros_like(alpha[None, :, None]))
                grad1 += kxz_gradx.sum(1)
                grad2 += kxz_gradx2.sum(1)

            return log_px, grad1, grad2
        elif mode == "CP":
            log_px, grad1, S_r, S_i = self.q0_S(x_v)
            for (kernel, kw) in zip(self.kernel, kernel_weight):
                kxz = kernel(x_v, self.z)
                log_px += kxz @ alpha * kw
                kxz_gradx, kxzS_r, kxzS_i = kernel.grads_backward_S(alpha[None, :] * kw,
                                                              torch.zeros_like(alpha[None, :]),
                                                              torch.zeros_like(alpha[None, :]))

                grad1 += kxz_gradx.sum(1)
                S_r += kxzS_r.sum(1)
                S_i += kxzS_i.sum(1)

            return log_px, grad1, S_r, S_i


class Kernel(nn.Module):
    def __init__(self, input_dim, hidden_dim, add_skip, sigma_init=1.0, pretrained_encoder=None):
        super().__init__()
        self.input_dim = input_dim
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init))
        self.phi_w = Phi_w(self.input_dim, hidden_dim=hidden_dim, add_skip=add_skip)

        if pretrained_encoder is None:
            def init_weights(m):
                if type(m) == nn.Linear:
                    m.weight.data = torch.tensor(np.random.randn(*list(m.weight.size())) / np.sqrt(m.weight.size(0))).float()
                    if m.bias is not None:
                        m.bias.data = torch.tensor(np.random.randn(*list(m.bias.size())) / np.sqrt(m.bias.size(0))).float()

            self.phi_w.apply(init_weights)
        else:
            self.phi_w.load_state_dict(pretrained_encoder)

    # NOTE: sigma is not squared to replicate code in the DKEF paper.
    # This does not meaningfully change the model.
    def forward(self, x, z):
        diff = ((self.phi_w(x)[:,None,:] - self.phi_w(z)) ** 2).sum(2)
        return torch.exp(-1/(2 * torch.pow(10.0, self.log_sigma)) * diff)

    def alpha_forward(self, x, z):
        diff = ((self.phi_w(x) - self.phi_w(z)) ** 2).sum(2)
        return torch.exp(-1/(2 * torch.pow(10.0, self.log_sigma)) * diff)


class Phi_w(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, add_skip=False):
        super().__init__()
        self.input_dim = input_dim
        self.add_skip = add_skip
        if add_skip: # Only have bias on the final hidden layer, not the skip.
            self.skip = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        if self.add_skip:
            return self.net(x) + self.skip(x)
        else:
            return self.net(x)

class Phi_w_approx_bp(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, add_skip=False):
        super().__init__()
        self.input_dim = input_dim
        self.add_skip = add_skip
        if add_skip: # Only have bias on the final hidden layer, not the skip.
            self.skip = models.nice_approxbp.Linear(self.input_dim, hidden_dim, bias=False)
        self.dense1 = models.nice_approxbp.Linear(self.input_dim, hidden_dim)
        self.act1 = models.nice_approxbp.Softplus()
        self.dense2 = models.nice_approxbp.Linear(hidden_dim, hidden_dim)
        self.act2 = models.nice_approxbp.Softplus()
        self.dense3 = models.nice_approxbp.Linear(hidden_dim, hidden_dim)

    def forward(self, x, save_grad):
        if self.add_skip:
            return self.dense3(
                self.act2(self.dense2(
                    self.act1(self.dense1(x), save_grad=save_grad)), save_grad=save_grad)) + self.skip(x)
        else:
            return self.dense3(
                self.act2(self.dense2(
                    self.act1(self.dense1(x), save_grad=save_grad)), save_grad=save_grad))

    def grads_backward(self, grad1, grad2):
        grad1_net, grad2_net = self.dense1.grads_backward(
            *self.act1.grads_backward(
                *self.dense2.grads_backward(
                    *self.act2.grads_backward(
                        *self.dense3.grads_backward(grad1, grad2)
                    )
                )
            )
        )
        if self.add_skip:
            grad1_skip, grad2_skip = self.skip.grads_backward(grad1, grad2)
            grad1_net += grad1_skip
            grad2_net += grad2_skip

        return grad1_net, grad2_net

    def grads_backward_S(self, grad1, S_r, S_i):
        grad1_net, S_rnet, S_inet = self.dense1.grads_backward_S(
            *self.act1.grads_backward_S(
                *self.dense2.grads_backward_S(
                    *self.act2.grads_backward_S(
                        *self.dense3.grads_backward_S(grad1, S_r, S_i)
                    )
                )
            )
        )
        if self.add_skip:
            grad1_skip, S_rskip, S_iskip = self.skip.grads_backward_S(grad1, S_r, S_i)
            grad1_net += grad1_skip
            S_rnet += S_rskip
            S_inet += S_iskip

        return grad1_net, S_rnet, S_inet

class Kernel_approx_bp(nn.Module):
    def __init__(self, input_dim, hidden_dim, add_skip, sigma_init=1.0, pretrained_encoder=None):
        super().__init__()
        self.input_dim = input_dim
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init))
        self.phi_w = Phi_w_approx_bp(self.input_dim, hidden_dim=hidden_dim, add_skip=add_skip)

        if pretrained_encoder is None:
            def init_weights(m):
                if type(m) == models.nice_approxbp.Linear:
                    m.weight.data = torch.tensor(np.random.randn(*list(m.weight.size())) / np.sqrt(m.weight.size(0))).float()
                    if m.bias is not None:
                        m.bias.data = torch.tensor(np.random.randn(*list(m.bias.size())) / np.sqrt(m.bias.size(0))).float()

            self.phi_w.apply(init_weights)
        else:
            self.phi_w.load_state_dict(pretrained_encoder)

    # NOTE: sigma is not squared to replicate code in the DKEF paper.
    # This does not meaningfully change the model.
    def forward(self, x, z):
        self.diff = (self.phi_w(x, save_grad=True)[:,None,:] - self.phi_w(z, save_grad=False))
        self.diffsq = (self.diff ** 2).sum(2)
        self.kxz = torch.exp(-1/(2 * torch.pow(10.0, self.log_sigma)) * self.diffsq)
        return self.kxz

    def grads_backward(self, grad1=None, grad2=None):
        scaled_diff = -1 / (torch.pow(10.0, self.log_sigma)) * self.diff
        self_grad1 = scaled_diff * self.kxz[:, :, None]
        self_grad2 = self.kxz[:, :, None] * (scaled_diff ** 2) - 1 / (torch.pow(10.0, self.log_sigma)) * self.kxz[:, :,
                                                                                                         None]

        if grad1 is None or grad2 is None:
            new_grad1, new_grad2 = self.phi_w.grads_backward(self_grad1, self_grad2)
        else:
            new_grad1 = grad1 * self_grad1
            new_grad2 = grad1 * self_grad2 + grad2 * self_grad1 ** 2
            new_grad1, new_grad2 = self.phi_w.grads_backward(new_grad1, new_grad2)

        return new_grad1, new_grad2

    def grads_backward_S(self, grad1=None, S_r=None, S_i=None):
        if grad1 is None and S_r is None and S_i is None:
            grad1 = torch.ones_like(self.kxz)
            S_r = torch.zeros_like(grad1)
            S_i = torch.zeros_like(S_r)

        # First, pass backwards through the exponential:
        v = torch.randn_like(self.kxz).sign()
        M = grad1 * self.kxz
        S_r = torch.sqrt(torch.relu(M)) * v + self.kxz * S_r
        S_i = torch.sqrt(torch.relu(-M)) * v + self.kxz * S_i
        grad1 = self.kxz * grad1

        # Then, backwards pass through the scaled norm
        grad1_normnode = -1 / (torch.pow(10.0, self.log_sigma)) * self.diff
        M = -grad1 / torch.pow(10.0, self.log_sigma)
        v = torch.randn_like(grad1_normnode).sign()
        S_r = torch.sqrt(torch.relu(M))[:, :, None] * v + grad1_normnode * S_r[:, :, None]
        S_i = torch.sqrt(torch.relu(-M))[:, :, None] * v + grad1_normnode * S_i[:, :, None]
        grad1 = grad1[:, :, None] * grad1_normnode

        # And finally through phi net
        new_grad1, new_S_r, new_S_i = self.phi_w.grads_backward_S(grad1, S_r, S_i)

        return new_grad1, new_S_r, new_S_i

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=30):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = Phi_w(input_dim, hidden_dim)
        self.final_layer = nn.Linear(self.hidden_dim, 10)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.final_layer(torch.nn.functional.softplus(encoded))

if __name__ == '__main__':
    import models.nice
    import torch.autograd as autograd

    test_dkef = True
    dim = 15
    if test_dkef:
        test_Z = torch.randn(200, 15)
        dkef = DKEF(15, "CP", num_kernels=1, init_z=test_Z, M=200,
                 sigma_list=(0.0, 0.5, 1.0), hidden_dim=30, add_skip=False,
                 alpha_param=False, train_Z=True, pretrained_encoder=None, dsm_sigma=None)

        for i in range(1):
            test_input_t = torch.randn(100, 15, requires_grad=True)
            test_input_v = torch.randn(100, 15, requires_grad=True)

            dkef.zero_alpha_matrices(test_input_t.device)
            dkef.accumulate_alpha_matrices(data=test_input_t, vectors=torch.randn_like(test_input_t).sign())
            dkef.update_alpha(None, stage="finalize")

            test_input_t = None
            output, grad1_kingma, grad2_kingma = dkef.approx_bp_forward(test_input_t, test_input_v, stage="finalize")
            grad_std = autograd.grad(output.sum(), test_input_v, create_graph=True)[0]
            grad2_std = torch.stack([autograd.grad(grad_std[:,d].sum(), test_input_v, retain_graph=True)[0][:,d] for d in range(dim)], dim=1)

            grad2_S_acc = torch.zeros_like(grad2_kingma)
            niters = 2000
            for _ in tqdm(range(niters)):
                output, grad1_S, S_r, S_i = dkef.approx_bp_forward(test_input_t, test_input_v, stage="finalize", mode="CP")
                grad2_S = S_r ** 2 - S_i ** 2
                grad2_S_acc += grad2_S.detach()
            grad2_S_acc /= niters

            print("Grad2 norm: ", ((grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 kingma: ", ((grad2_kingma - grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 S: ", ((grad2_S_acc - grad2_std) ** 2).sum(1).mean())

            print('#{}, diff in grad S: {}, diff in grad kingma: {}'.format(
                i + 1,
                (torch.norm(grad_std - grad1_S, dim=-1).mean() / torch.norm(grad_std, dim=-1).mean()).item(),
                (torch.norm(grad_std - grad1_kingma, dim=-1).mean() / torch.norm(grad_std, dim=-1).mean()).item(),
            ))

    test_kernel = False
    if test_kernel:
        np.random.seed(0)
        kernel_std = Kernel(15, 30, False)
        np.random.seed(0)
        kernel = Kernel_approx_bp(15, 30, False)

        for i in range(1):
            test_input = torch.randn(100, 15, requires_grad=True)
            expanded_test_input = test_input[:,None,:].expand(-1, 200, -1)
            test_Z = torch.randn(200, 15)
            energy_std = kernel_std.alpha_forward(expanded_test_input, test_Z)
            energy = kernel.forward(test_input, test_Z)

            grad_std = autograd.grad(energy_std.sum(), expanded_test_input, create_graph=True)[0]
            grad2_std = torch.stack([autograd.grad(grad_std[:,:,d].sum(), expanded_test_input, retain_graph=True)[0][:,:,d] for d in range(dim)], dim=2)
            grad1_kingma, grad2_kingma = kernel.grads_backward()
            grad2_S_acc = torch.zeros_like(grad2_kingma)
            niters = 10
            for _ in range(niters):
                grad1_S, S_r, S_i = kernel.grads_backward_S(grad1=torch.ones_like(energy),
                                                            S_r=torch.zeros_like(energy), S_i=torch.zeros_like(energy))
                grad2_S = S_r ** 2 - S_i ** 2
                grad2_S_acc += grad2_S.detach()
            grad2_S_acc /= niters

            grad2_hutch_acc = torch.zeros_like(grad2_kingma)
            for _ in range(niters):
                v = torch.randn_like(grad_std).sign()
                grad1v = grad_std * v
                grad2v = torch.autograd.grad(grad1v.sum(), expanded_test_input, retain_graph=True)[0]
                grad2_hutch = grad2v * v
                grad2_hutch_acc += grad2_hutch
            grad2_hutch_acc /= niters

            print("Grad2 norm: ", ((grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 kingma: ", ((grad2_kingma - grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 S: ", ((grad2_S_acc - grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 hutch: ", ((grad2_hutch_acc - grad2_std) ** 2).sum(1).mean())

            print('#{}, diff in energy: {}, diff in grad: {}'.format(
                i + 1,
                torch.norm(energy_std - energy, dim=-1).mean().item(),
                torch.norm(grad_std - grad1_S, dim=-1).mean().item(),
            ))

    test_phi = False
    if test_phi:
        np.random.seed(0)
        phi_std = Phi_w(dim, 30, False)
        phi = Phi_w_approx_bp(dim, 30, False)
        def init_weights(m):
            np.random.seed(0)
            if type(m) == models.nice_approxbp.Linear or type(m) == nn.Linear:
                m.weight.data = torch.tensor(
                    np.random.randn(*list(m.weight.size())) / np.sqrt(m.weight.size(0))).float()
                if m.bias is not None:
                    m.bias.data = torch.tensor(np.random.randn(*list(m.bias.size())) / np.sqrt(m.bias.size(0))).float()
        phi_std.apply(init_weights)
        phi.apply(init_weights)

        for i in range(1):
            test_input = torch.randn(100, dim, requires_grad=True)
            phi_x_std = phi_std.forward(test_input)
            phi_x = phi.forward(test_input, save_grad=True)

            grad_std = autograd.grad(phi_x_std.sum(), test_input, create_graph=True)[0]
            grad2_std = torch.stack([autograd.grad(grad_std[:,d].sum(), test_input, retain_graph=True)[0][:,d] for d in range(dim)], dim=1)
            grad1_kingma, grad2_kingma = phi.grads_backward(torch.ones(30), torch.zeros(30))
            grad2_S_acc = torch.zeros_like(grad2_kingma)
            niters = 200
            for _ in range(niters):
                grad1_S, S_r, S_i = phi.grads_backward_S(torch.ones(30), torch.zeros(30), torch.zeros(30))
                grad2_S = S_r ** 2 - S_i ** 2
                grad2_S_acc += grad2_S.detach()
            grad2_S_acc /= niters

            print("Grad2 norm: ", ((grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2: ", ((grad2_kingma - grad2_std) ** 2).sum(1).mean())
            print("Diff in grad2 S: ", ((grad2_S_acc - grad2_std) ** 2).sum(1).mean())

            print('#{}, diff in energy: {}, diff in grad: {}'.format(
                i + 1,
                torch.norm(phi_x_std - phi_x, dim=-1).mean().item(),
                torch.norm(grad_std - grad1_kingma, dim=-1).mean().item(),
            ))