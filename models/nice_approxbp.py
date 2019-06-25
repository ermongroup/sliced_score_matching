import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import torch.autograd as autograd

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            logdet = - inputs - 2. * F.softplus(-inputs)
            return s(inputs), logdet.sum(-1, keepdim=True)
        else:
            logdet = -torch.log(inputs) - torch.log(1. - inputs)
            return torch.log(inputs) - torch.log(1 - inputs), logdet.sum(-1, keepdim=True)

class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, mode='inverse')
        else:
            return super(Logit, self).forward(inputs, mode='direct')

class Linear(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        self.grad = self.weight.t()
        return output

    def grads_backward(self, grad1, grad2):
        new_grad1 = F.linear(grad1, self.grad, bias=None)
        new_grad2 = F.linear(grad2, self.grad ** 2, bias=None)
        return new_grad1, new_grad2

    def grads_backward_TU(self, grad1, T, U):
        T = F.linear(T, self.grad, bias=None)
        v = torch.randn_like(T).sign()
        U = v + F.linear(U, self.grad, bias=None)
        new_grad1 = F.linear(grad1, self.grad, bias=None)
        return new_grad1, T, U

    def grads_backward_S(self, grad1, S_r, S_i):
        S_r = F.linear(S_r, self.grad, bias=None)
        S_i = F.linear(S_i, self.grad, bias=None)
        new_grad1 = F.linear(grad1, self.grad, bias=None)
        return new_grad1, S_r, S_i

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Softplus(nn.Module):
    __constants__ = ['beta', 'threshold']

    def __init__(self, beta=1, threshold=20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input, save_grad=True):
        output = F.softplus(input, self.beta, self.threshold)
        if save_grad:
            self.grad1 = torch.sigmoid(input)
            self.grad2 = torch.sigmoid(input) * (1. - torch.sigmoid(input))
        return output

    def grads_backward(self, grad1, grad2):
        if len(grad1.shape) == 3 and len(self.grad1.shape) == 2:
            self.grad1 = self.grad1[:,None,:]
            self.grad2 = self.grad2[:,None,:]

        new_grad1 = grad1 * self.grad1
        new_grad2 = grad1 * self.grad2 + grad2 * self.grad1 ** 2
        return new_grad1, new_grad2

    def grads_backward_TU(self, grad1, T, U):
        # print("Entering Softplus")
        # print((grad1.shape, T.shape, U.shape))
        M = self.grad2 * grad1 # Note, here M is a vector instead of a diagonal matrix
        T = T * self.grad1
        v = torch.randn_like(T).sign()
        T = M * v + T
        U = v +  U  * self.grad1
        new_grad1 = grad1 * self.grad1
        # print((new_grad1.shape, T.shape, U.shape))
        return new_grad1, T, U

    def grads_backward_S(self, grad1, S_r, S_i):
        if len(grad1.shape) == 3 and len(self.grad1.shape) == 2:
            self.grad1 = self.grad1[:,None,:]
            self.grad2 = self.grad2[:,None,:]

        M = self.grad2 * grad1 # Note, here M is a vector instead of a diagonal matrix
        F_r = torch.sqrt(torch.relu(M))
        F_i = torch.sqrt(torch.relu(-M))
        v = torch.randn_like(F_r).sign()
        S_r = F_r * v + S_r * self.grad1
        S_i = F_i * v + S_i * self.grad1
        new_grad1 = grad1 * self.grad1
        return new_grad1, S_r, S_i

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class NiceScaleLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.scales = nn.Parameter(torch.zeros(1, size))

    def forward(self, X, inv=False):
        if not inv:
            self.grad1 = torch.exp(self.scales)
            self.grad2 = 0.

            return X * torch.exp(self.scales), torch.sum(self.scales)
        else:
            self.grad1 = self.grad2 = None
            return X * torch.exp(-self.scales), torch.sum(-self.scales)

    def grads_backward(self, grad1, grad2):
        new_grad1 = grad1 * self.grad1
        new_grad2 = grad1 * self.grad2 + grad2 * self.grad1 ** 2
        return new_grad1, new_grad2

    def grads_backward_TU(self, grad1, T, U):
        new_grad1 = grad1 * self.grad1
        T = T * self.grad1
        v = torch.randn_like(T).sign()
        U = v + U * self.grad1
        return new_grad1, T, U

    def grads_backward_S(self, grad1, S_r, S_i):
        new_grad1 = grad1 * self.grad1
        S_r = S_r * self.grad1
        S_i = S_i * self.grad1
        return new_grad1, S_r, S_i

    def inv_scale_jac(self):
        return torch.sum(-self.scales)


class NiceLayer(nn.Module):
    # Note: only support num_layers=2, with tanh (as in OLDNICE)
    # or num_layers=5 with relu (as in NICEPAPER)
    def __init__(self, size, hidden_size, num_layers=2):
        super().__init__()
        self.half_size = half_size = size // 2
        self.num_layers = num_layers
        self.dense1 = Linear(half_size, hidden_size)
        self.act1 = Softplus()
        if self.num_layers == 2:
            self.dense2 = Linear(hidden_size, half_size)

        elif self.num_layers == 5:
            self.dense2 = Linear(hidden_size, hidden_size)
            self.act2 = Softplus()
            self.dense3 = Linear(hidden_size, hidden_size)
            self.act3 = Softplus()
            self.dense4 = Linear(hidden_size, hidden_size)
            self.act4 = Softplus()
            self.dense5 = Linear(hidden_size, half_size)
        else:
            raise ValueError("Only supports 2 or 5 layers in a coupling layer")

    def _m_net(self, X):
        if self.num_layers == 2:
            l1 = self.act1(self.dense1(X))
            return self.dense2(l1)
        else:
            l1 = self.act1(self.dense1(X))
            l2 = self.act2(self.dense2(l1))
            l3 = self.act3(self.dense3(l2))
            l4 = self.act4(self.dense4(l3))
            l5 = self.dense5(l4)
            return l5

    def _m_net_grads_backward(self, grad1, grad2):
        if self.num_layers == 2:
            grad1, grad2 = self.dense1.grads_backward(
                *self.act1.grads_backward(*self.dense2.grads_backward(grad1, grad2)))
        else:
            grad1, grad2 = self.dense1.grads_backward(
                *self.act1.grads_backward(
                    *self.dense2.grads_backward(
                        *self.act2.grads_backward(
                            *self.dense3.grads_backward(
                                *self.act3.grads_backward(
                                    *self.dense4.grads_backward(
                                        *self.act4.grads_backward(
                                            *self.dense5.grads_backward(grad1, grad2)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        return grad1, grad2

    def _m_net_grads_backward_TU(self, grad1, T, U):
        if self.num_layers == 2:
            grad1, T, U = self.dense1.grads_backward_TU(
                *self.act1.grads_backward_TU(*self.dense2.grads_backward_TU(grad1, T, U)))
        else:
            grad1, T, U = self.dense1.grads_backward_TU(
                *self.act1.grads_backward_TU(
                    *self.dense2.grads_backward_TU(
                        *self.act2.grads_backward_TU(
                            *self.dense3.grads_backward_TU(
                                *self.act3.grads_backward_TU(
                                    *self.dense4.grads_backward_TU(
                                        *self.act4.grads_backward_TU(
                                            *self.dense5.grads_backward_TU(grad1, T, U)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        return grad1, T, U

    def _m_net_grads_backward_S(self, grad1, S_r, S_i):
        if self.num_layers == 2:
            grad1, S_r, S_i = self.dense1.grads_backward_S(
                *self.act1.grads_backward_S(*self.dense2.grads_backward_S(grad1, S_r, S_i)))
        else:
            grad1, S_r, S_i = self.dense1.grads_backward_S(
                *self.act1.grads_backward_S(
                    *self.dense2.grads_backward_S(
                        *self.act2.grads_backward_S(
                            *self.dense3.grads_backward_S(
                                *self.act3.grads_backward_S(
                                    *self.dense4.grads_backward_S(
                                        *self.act4.grads_backward_S(
                                            *self.dense5.grads_backward_S(grad1, S_r, S_i)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        return grad1, S_r, S_i


    def forward(self, X, type=0, inv=False):
        x1 = X[:, :self.half_size]
        x2 = X[:, self.half_size:]
        if type == 0:
            m1 = self._m_net(x1)
            delta = torch.cat([torch.zeros_like(x1), m1], dim=1)
        elif type == 1:
            m2 = self._m_net(x2)
            delta = torch.cat([m2, torch.zeros_like(x2)], dim=1)

        if not inv:
            return X + delta
        else:
            return X - delta

    def grads_backward(self, grad1, grad2, type=0, inv=False):
        if inv:
            return None, None

        if type == 0:
            gradm1 = grad1[:, self.half_size:]
            gradm2 = grad2[:, self.half_size:]
            gradm1, gradm2 = self._m_net_grads_backward(gradm1, gradm2)
            gradm1 = torch.cat([gradm1, torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device)], dim=1)
            gradm2 = torch.cat([gradm2, torch.zeros(gradm2.shape[0], self.half_size, device=grad1.device)], dim=1)

        elif type == 1:
            gradm1 = grad1[:, :self.half_size]
            gradm2 = grad2[:, :self.half_size]
            gradm1, gradm2 = self._m_net_grads_backward(gradm1, gradm2)
            gradm1 = torch.cat([torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device), gradm1], dim=1)
            gradm2 = torch.cat([torch.zeros(gradm2.shape[0], self.half_size, device=grad1.device), gradm2], dim=1)

        return grad1 + gradm1, grad2 + gradm2

    def grads_backward_TU(self, grad1, T, U, type=0, inv=False):
        if inv:
            return None, None

        if type == 0:
            gradm1 = grad1[:, self.half_size:]
            Tm = T[:, self.half_size:]
            Um = U[:, self.half_size:]
            gradm1, Tm, Um = self._m_net_grads_backward_TU(gradm1, Tm, Um)
            gradm1 = torch.cat([gradm1, torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device)], dim=1)
            Tm = torch.cat([Tm, torch.zeros(Tm.shape[0], self.half_size, device=grad1.device)], dim=1)
            Um = torch.cat([Um, torch.zeros(Um.shape[0], self.half_size, device=grad1.device)], dim=1)
        if type == 1:
            gradm1 = grad1[:, :self.half_size]
            Tm = T[:, :self.half_size]
            Um = U[:, :self.half_size]
            gradm1, Tm, Um = self._m_net_grads_backward_TU(gradm1, Tm, Um)
            gradm1 = torch.cat([torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device), gradm1], dim=1)
            Tm = torch.cat([torch.zeros(Tm.shape[0], self.half_size, device=grad1.device), Tm], dim=1)
            Um = torch.cat([torch.zeros(Um.shape[0], self.half_size, device=grad1.device), Um], dim=1)

        return grad1 + gradm1, T + Tm, U + Um

    def grads_backward_S(self, grad1, S_r, S_i, type=0, inv=False):
        if inv:
            return None, None

        if type == 0:
            gradm1 = grad1[:, self.half_size:]
            S_rm = S_r[:, self.half_size:]
            S_im = S_i[:, self.half_size:]
            gradm1, S_rm, S_im = self._m_net_grads_backward_S(gradm1, S_rm, S_im)
            gradm1 = torch.cat([gradm1, torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device)], dim=1)
            S_rm = torch.cat([S_rm, torch.zeros(S_rm.shape[0], self.half_size, device=grad1.device)], dim=1)
            S_im = torch.cat([S_im, torch.zeros(S_im.shape[0], self.half_size, device=grad1.device)], dim=1)
        if type == 0:
            gradm1 = grad1[:, :self.half_size]
            S_rm = S_r[:, :self.half_size]
            S_im = S_i[:, :self.half_size]
            gradm1, S_rm, S_im = self._m_net_grads_backward_S(gradm1, S_rm, S_im)
            gradm1 = torch.cat([torch.zeros(gradm1.shape[0], self.half_size, device=grad1.device), gradm1], dim=1)
            S_rm = torch.cat([torch.zeros(S_rm.shape[0], self.half_size, device=grad1.device), S_rm], dim=1)
            S_im = torch.cat([torch.zeros(S_im.shape[0], self.half_size, device=grad1.device), S_im], dim=1)

        return grad1 + gradm1, S_r + S_rm, S_i + S_im

class NICE(nn.Module):
    def __init__(self, size, hidden_nodes, num_layers=2):
        super().__init__()
        act = nn.Softplus()

        self.nice1 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice2 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice3 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice4 = NiceLayer(size, hidden_nodes, num_layers)
        self.scale = NiceScaleLayer(size)

    def _permutate(self, tensor, neurons, inv=False):
        permutation = np.arange(0, neurons)
        perm = permutation.copy()
        perm[:len(permutation) // 2] = permutation[::2]
        perm[len(permutation) // 2:] = permutation[1::2]
        inv_perm = np.argsort(perm)
        if not inv:
            to_perm = torch.from_numpy(np.identity(len(permutation))[:, perm]).to(tensor.device).type_as(tensor)
            return tensor @ to_perm
        else:
            inv_perm = torch.from_numpy(np.identity(len(permutation))[:, inv_perm]).to(tensor.device).type_as(tensor)
            return tensor @ inv_perm

    def forward(self, X, inv=False):
        if not inv:
            y = self._permutate(X, X.shape[1], inv=inv)
            jac_pre = 0.0
            y = self.nice1(y, type=0, inv=inv)
            y = self.nice2(y, type=1, inv=inv)
            y = self.nice3(y, type=0, inv=inv)
            y = self.nice4(y, type=1, inv=inv)

            y, jac = self.scale(y, inv=inv)
            dim = y.shape[1]
            self.output = y
            return -torch.tensor(dim * 0.5 * np.log(2 * np.pi), device=X.device) \
                   - 0.5 * torch.sum(y ** 2, dim=1) + jac + jac_pre, y
        else:
            y, jac = self.scale(X, inv=inv)
            dim = y.shape[1]
            y = self.nice4(y, type=1, inv=inv)
            y = self.nice3(y, type=0, inv=inv)
            y = self.nice2(y, type=1, inv=inv)
            y = self.nice1(y, type=0, inv=inv)
            return self._permutate(y, X.shape[1], inv=inv), \
                   -torch.tensor(dim * 0.5 * np.log(2 * np.pi), device=X.device) - 0.5 * torch.sum(X ** 2, dim=1) - jac

    def grads_backward(self, inv=False):
        if not inv:
            grad1 = -self.output
            grad2 = -torch.ones_like(self.output)
            grad1, grad2 = self.nice1.grads_backward(
                *self.nice2.grads_backward(
                    *self.nice3.grads_backward(
                        *self.nice4.grads_backward(
                            *self.scale.grads_backward(grad1, grad2)
                        )
                    )
                )
            )

            grad1 = self._permutate(grad1, grad1.shape[1], inv=not inv)
            grad2 = self._permutate(grad2, grad2.shape[1], inv=not inv)
            return grad1, grad2
        else:
            return None, None

    def grads_backward_TU(self, inv=False):
        if not inv:
            grad1 = -self.output
            M_diag = -torch.ones_like(self.output)
            v = torch.randn_like(self.output).sign()
            T = M_diag * v
            U = v

            grad1, T, U = self.nice1.grads_backward_TU(
                *self.nice2.grads_backward_TU(
                    *self.nice3.grads_backward_TU(
                        *self.nice4.grads_backward_TU(
                            *self.scale.grads_backward_TU(grad1, T, U)
                        )
                    )
                )
            )

            grad1 = self._permutate(grad1, grad1.shape[1], inv=not inv)
            T = self._permutate(T, T.shape[1], inv=not inv)
            U = self._permutate(U, U.shape[1], inv=not inv)
            return grad1, T, U
        else:
            return None, None

    def grads_backward_S(self, inv=False):
        if not inv:
            grad1 = -self.output
            S_r = torch.zeros_like(self.output)
            S_i = torch.randn_like(self.output).sign() * torch.ones_like(self.output)

            grad1, S_r, S_i = self.nice1.grads_backward_S(
                *self.nice2.grads_backward_S(
                    *self.nice3.grads_backward_S(
                        *self.nice4.grads_backward_S(
                            *self.scale.grads_backward_S(grad1, S_r, S_i)
                        )
                    )
                )
            )

            grad1 = self._permutate(grad1, grad1.shape[1], inv=not inv)
            S_r = self._permutate(S_r, S_r.shape[1], inv=not inv)
            S_i = self._permutate(S_i, S_i.shape[1], inv=not inv)
            return grad1, S_r, S_i
        else:
            return None, None

    def inv_scale_jac(self):
        return self.scale.inv_scale_jac()


class ToyNet(nn.Module):
    def __init__(self, input_dim=2, hidden_units=32):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_units)
        self.act1 = Softplus()
        self.fc2 = Linear(hidden_units, hidden_units)
        self.act2 = Softplus()
        self.fc3 = Linear(hidden_units, 1)

    def forward(self, inputs):
        y = self.act1(self.fc1(inputs))
        y = self.act2(self.fc2(y))
        y = self.fc3(y)
        self.output = y
        return y

    def grads_backward(self):
        grad1 = torch.ones_like(self.output)
        grad2 = torch.zeros_like(self.output)
        grad1, grad2 = self.fc1.grads_backward(
            *self.act1.grads_backward(
                *self.fc2.grads_backward(
                    *self.act2.grads_backward(
                        *self.fc3.grads_backward(grad1, grad2)
                    )
                )
            )
        )
        return grad1, grad2

    def grads_backward_TU(self):
        T = torch.zeros_like(self.output)
        U = torch.zeros_like(self.output)
        grad1 = torch.ones_like(self.output)

        grad1, T, U = self.fc1.grads_backward_TU(
            *self.act1.grads_backward_TU(
                *self.fc2.grads_backward_TU(
                    *self.act2.grads_backward_TU(
                        *self.fc3.grads_backward_TU(grad1, T, U)
                    )
                )
            )
        )
        return grad1, T, U

    def grads_backward_S(self):
        S_r = torch.zeros_like(self.output)
        S_i = torch.zeros_like(self.output)
        grad1 = torch.ones_like(self.output)

        grad1, S_r, S_i = self.fc1.grads_backward_S(
            *self.act1.grads_backward_S(
                *self.fc2.grads_backward_S(
                    *self.act2.grads_backward_S(
                        *self.fc3.grads_backward_S(grad1, S_r, S_i)
                    )
                )
            )
        )
        return grad1, S_r, S_i


class ShallowNet(nn.Module):
    def __init__(self, input_dim=2, hidden_units=32):
        super().__init__()
        self.fc1 = Linear(input_dim, 1)
        self.act1 = Softplus()

    def forward(self, inputs):
        y = self.act1(self.fc1(inputs))
        self.output = y
        return y

    def grads_backward(self):
        grad1 = torch.ones_like(self.output)
        grad2 = torch.zeros_like(self.output)
        grad1, grad2 = self.fc1.grads_backward(
            *self.act1.grads_backward(grad1, grad2)
        )
        return grad1, grad2

    def grads_backward_TU(self):
        T = torch.zeros_like(self.output)
        U = torch.zeros_like(self.output)
        grad1 = torch.ones_like(self.output)

        grad1, T, U = self.fc1.grads_backward_TU(
            *self.act1.grads_backward_TU(grad1, T, U)
        )
        return grad1, T, U

def approx_backprop_score_matching(grad_net, samples):
    grad1, grad2 = grad_net(samples)
    grad1 = -grad1
    grad2 = -grad2
    loss = 0.5 * grad1.norm(dim=-1) ** 2 + grad2.sum(dim=-1)
    return loss.mean()

# test the implementation of grads_backward()
if __name__ == '__main__':
    import models.nice
    import torch.autograd as autograd

    test_nice = False
    if test_nice:
        nice_std = models.nice.NICE(784, 128, 5)
        nice = NICE(784, 128, 5)
        nice.load_state_dict(nice_std.state_dict())

        for i in range(1):
            test_input = torch.randn(100, 784, requires_grad=True)
            energy_std, _ = nice_std(test_input, inv=False)
            energy, y = nice(test_input, inv=False)

            grad_std = autograd.grad(energy_std.sum(), test_input)[0]
            grad1, grad2 = nice.grads_backward(inv=False)

            recon, _ = nice(y, inv=True)
            print('#{}, diff in energy: {}, diff in grad: {}, recon: {}'.format(
                i + 1,
                torch.norm(energy_std - energy, dim=-1).mean().item(),
                torch.norm(grad_std - grad1, dim=-1).mean().item(),
                torch.norm(recon - test_input, dim=-1).mean().item()
            ))
    test_toy_net = False
    if test_toy_net:
        model = ToyNet(2, 32)
        for i in range(1000):
            test_input = torch.randn(100, 2, requires_grad=True)
            energy = model(test_input)
            grad_std = autograd.grad(energy.sum(), test_input)[0]
            grad1, grad2 = model.grads_backward()
            print('#{}, diff in grad: {}'.format(i + 1, torch.norm(grad_std - grad1, dim=-1).mean().item()))

    test_UT = True
    if test_UT:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dim = 20
        model = ToyNet(dim, 200).to(device)
        means = torch.randn(dim).to(device) * 0
        stdevs = torch.ones(dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in range(0):
            test_input = torch.randn(100, dim, requires_grad=True).to(device) * stdevs + means
            logp, _ = model(test_input)
            loss = -logp.mean()
            optimizer.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()

        nsamples = 100
        for i in range(1):
            test_input = torch.randn(nsamples, dim, requires_grad=True).to(device) * stdevs + means
            energy = model(test_input)
            grad_std = autograd.grad(energy.sum(), test_input, create_graph=True, retain_graph=True)[0]
            grad2_std = torch.stack([autograd.grad(grad_std[:,d].sum(), test_input, retain_graph=True)[0] for d in range(dim)], dim=1)
            grad2_std_diag = torch.stack([autograd.grad(grad_std[:,d].sum(), test_input, retain_graph=True)[0][:,d] for d in range(dim)], dim=1)
            grad2_hutch_acc = torch.zeros(nsamples, dim, dim).to(device)

            n_iters = 1
            for _ in range(n_iters):
                v = torch.randn_like(grad_std).sign()
                grad2v = autograd.grad((grad_std * v).sum(), test_input, retain_graph=True)[0]
                grad2_hutch = grad2v[:,:,None] * v[:, None, :]
                grad2_hutch_acc += grad2_hutch.detach()
            grad2_hutch_acc /= n_iters

            grad2_TU_acc = torch.zeros(nsamples, dim, dim).to(device)
            for _ in range(n_iters):
                grad1, T, U = model.grads_backward_TU()
                grad2 = (T[:,:,None] * U[:,None,:] + U[:,:,None] * T[:,None,:]) / 2.
                grad2_TU_acc += grad2.detach()
            grad2_TU_acc /= n_iters

            grad2_S_acc = torch.zeros(nsamples, dim, dim).to(device)
            for _ in range(n_iters):
                grad1, S_r, S_i = model.grads_backward_S()
                grad2 = (S_r[:,:,None] * S_r[:,None,:] - S_i[:,:,None] * S_i[:,None,:])
                grad2_S_acc += grad2.detach()
            grad2_S_acc /= n_iters


            grad1_kingma, grad2_kingma = model.grads_backward()

            print(grad2_kingma[0])
            print(grad2_std_diag[0])

            grad2_hutch_acc_diag = torch.stack([torch.diag(grad2_hutch_acc[i]) for i in range(nsamples)], dim=0)
            grad2_TU_acc_diag = torch.stack([torch.diag(grad2_TU_acc[i]) for i in range(nsamples)], dim=0)
            grad2_S_acc_diag = torch.stack([torch.diag(grad2_S_acc[i]) for i in range(nsamples)], dim=0)

            print("#" * 80)
            print("Diagonal evaluation")
            print("grad 2 std sum sq", (grad2_std_diag ** 2).sum(1).mean())
            print("kingma - std sum sq",((grad2_kingma - grad2_std_diag) ** 2).sum(1).mean())
            print("grad hutch - std sum sq",((grad2_hutch_acc_diag - grad2_std_diag) ** 2).sum(1).mean())
            print("grad UT - std sum sq",((grad2_TU_acc_diag - grad2_std_diag) ** 2).sum(1).mean())
            print("grad S - std sum sq",((grad2_S_acc_diag - grad2_std_diag) ** 2).sum(1).mean())

            print("#" * 80)
            print("Averaged diagonal evaluation")
            print("grad 2 std sum sq", ((grad2_std_diag ** 2).mean(0)).sum(0))
            print("kingma - std sum sq",((grad2_kingma - grad2_std_diag).mean(0) ** 2).sum(0))
            print("grad hutch - std sum sq",((grad2_hutch_acc_diag - grad2_std_diag).mean(0) ** 2).sum(0))
            print("grad UT - std sum sq",((grad2_TU_acc_diag - grad2_std_diag).mean(0) ** 2).sum(0))
            print("grad S - std sum sq",((grad2_S_acc_diag - grad2_std_diag).mean(0) ** 2).sum(0))

            print("#" * 80)
            print("Trace evaluation")
            tr_est_std = grad2_std_diag.sum(1)
            tr_est_kingma = grad2_kingma.sum(1)
            tr_est_hutch = grad2_hutch_acc_diag.sum(1)
            tr_est_TU = grad2_TU_acc_diag.sum(1)
            tr_est_S = grad2_S_acc_diag.sum(1)
            print("Tr est std", torch.norm(tr_est_std) ** 2 )
            print("Tr est std - kingma", torch.norm(tr_est_std - tr_est_kingma) ** 2)
            print("Tr est std - hutch", torch.norm(tr_est_std - tr_est_hutch) ** 2)
            print("Tr est std - TU", torch.norm(tr_est_std - tr_est_TU) ** 2)
            print("Tr est std - S", torch.norm(tr_est_std - tr_est_S) ** 2)

            print("#" * 80)
            print('#{}, diff in grad: {}'.format(i + 1, torch.norm(grad_std - grad1, dim=-1).mean().item()))