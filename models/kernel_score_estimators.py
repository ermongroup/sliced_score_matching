import torch
import numpy as np
import tensorflow as tf


class ScoreEstimator(object):
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return torch.exp(-torch.sum((x1 - x2) ** 2, dim=-1) /
                         (2 * kernel_width ** 2))

    def gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, dim=-2)
        x_col = torch.unsqueeze(x2, dim=-3)
        kernel_width = kernel_width[..., None, None]
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, dim=-2)
        x_col = torch.unsqueeze(x2, dim=-3)
        kernel_width = kernel_width[..., None, None]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width[..., None] ** 2)
        G_expand = torch.unsqueeze(G, dim=-1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        n_samples = x_samples.shape[-2]
        n_basis = x_basis.shape[-2]
        x_samples_expand = torch.unsqueeze(x_samples, dim=-2)
        x_basis_expand = torch.unsqueeze(x_basis, dim=-3)
        pairwise_dist = torch.sqrt(
            torch.sum((x_samples_expand - x_basis_expand) ** 2,
                      dim=-1))
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(
            pairwise_dist.view(-1, n_samples * n_basis),
            k=k, dim=-1)[0]
        kernel_width = top_k_values[:, -1].view(*(x_samples.shape[:-2]))
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()


class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=0.99):
        super().__init__()
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.enable_eager_execution(config=config)

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        # eigen_vectors: [..., M, n_eigen]
        # eigen_values: [..., n_eigen]
        # return: [..., N, n_eigen], by default n_eigen=M.
        M = samples.shape[-2]
        # Kxq: [..., N, M]
        # grad_Kx: [..., N, M, x_dim]
        # grad_Kq: [..., N, M, x_dim]
        Kxq = self.gram(x, samples, kernel_width)
        # Kxq = tf.Print(Kxq, [tf.shape(Kxq)], message="Kxq:")
        # ret: [..., N, n_eigen]
        ret = np.sqrt(M) * torch.matmul(Kxq, eigen_vectors)
        ret *= 1. / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            # _samples: [..., N + M, x_dim]
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.shape[-2]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M, device=samples.device)

        with tf.device("/cpu:0"):
            eigen_values, eigen_vectors = tf.self_adjoint_eig(Kq.cpu().numpy())

        eigen_values = torch.tensor(eigen_values.numpy(), device=samples.device)
        eigen_vectors = torch.tensor(eigen_vectors.numpy(), device=samples.device)

        # eigen_values = []
        # eigen_vectors = []
        # for Mat in Kq:
        #     e, v = torch.symeig(Mat, eigenvectors=True, upper=False)
        #     eigen_values.append(e)
        #     eigen_vectors.append(v)
        #
        # eigen_values = torch.stack(eigen_values, dim=0)
        # eigen_vectors = torch.stack(eigen_vectors, dim=0)

        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(
                eigen_values.view(-1, M), dim=0)
            # eigen_arr = eigen_arr[..., ::-1]
            eigen_arr = eigen_arr.flip([-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            self._n_eigen = torch.sum(
                (eigen_cum < self._n_eigen_threshold).int())
        if self._n_eigen is not None:
            # eigen_values: [..., n_eigen]
            # eigen_vectors: [..., M, n_eigen]
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(
            samples, x, eigen_vectors, eigen_values, kernel_width)
        # grad_K1_avg = [..., M, x_dim]
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        # beta: [..., n_eigen, x_dim]
        beta = -np.sqrt(M) * torch.matmul(
            eigen_vectors.transpose(-1, -2), grad_K1_avg) / torch.unsqueeze(
            eigen_values, dim=-1)
        # grads: [..., N, x_dim]
        grads = torch.matmul(eigen_ext, beta)
        return grads


class SteinScoreEstimator(ScoreEstimator):
    def __init__(self, eta=0.001):
        super().__init__()
        self._eta = eta

    def compute_gradients(self, samples, x=None):
        # samples: [..., M, x_dim]
        # x: [..., 1, x_dim]
        M = samples.shape[-2]
        # kernel_width: [...]
        kernel_width = self.heuristic_kernel_width(samples, samples)
        # K: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        K, grad_K1, grad_K2 = self.grad_gram(samples, samples,
                                             kernel_width)
        # K_inv: [..., M, M]
        Kinv = torch.inverse(K + self._eta * torch.eye(M, device=samples.device))
        # H_dh: [..., M, x_dim]
        H_dh = torch.sum(grad_K2, dim=-2)
        # grads: [..., M, x_dim]
        grads = - torch.matmul(Kinv, H_dh)
        if x is None:
            return grads
        else:
            assert x.shape[-2] == 1, "Only support single-particle out-of-sample extension."
            Kxx = self.gram(x, x, kernel_width)
            # Kxq: [..., 1, M]
            Kxq = self.gram(x, samples, kernel_width)
            # Kxq @ K_inv: [..., 1, M]
            KxqKinv = torch.matmul(Kxq, Kinv)
            # term1: [..., 1, 1]
            term1 = -1. / (Kxx + self._eta -
                           torch.matmul(KxqKinv, Kxq.transpose(-1, -2)))
            # grad_Kqx2: [..., M, 1, x_dim]
            Kqx, grad_Kqx1, grad_Kqx2 = self.grad_gram(samples, x, kernel_width)
            # term2: [..., 1, x_dim]
            term2 = torch.matmul(Kxq, grads) - torch.matmul(KxqKinv + 1.,
                                                            torch.squeeze(grad_Kqx2, -2))
            # ret: [..., 1, x_dim]
            return torch.matmul(term1, term2)
