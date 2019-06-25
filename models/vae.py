import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.mean = nn.Linear(256, self.z_dim)
        self.log_std = nn.Linear(256, self.z_dim)

    def forward(self, inputs):
        h = self.main(inputs.view(inputs.shape[0], -1))
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std


class MLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.main = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 784)
        )

    def forward(self, inputs):
        return self.main(inputs.view(inputs.shape[0], -1)).view(inputs.shape[0],
                                                                self.channels,
                                                                self.image_size,
                                                                self.image_size)


class MLPScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.main = nn.Sequential(
            nn.Linear(784 + self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X, z):
        X = X.view(X.shape[0], -1)
        Xz = torch.cat([X, z], dim=-1)
        h = self.main(Xz)
        return h


class MLPImplicitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps_dim = config.model.eps_dim
        self.z_dim = config.model.z_dim
        self.main = nn.Sequential(
            nn.Linear(784 + self.eps_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X):
        eps = torch.randn(X.shape[0], self.eps_dim, device=X.device)
        flattened_X = X.view(X.shape[0], -1)
        X_eps = torch.cat([flattened_X, eps], dim=-1)
        z = self.main(X_eps)
        return z


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.nef * 1, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Sequential(
            nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
            nn.ReLU(inplace=True)
        )
        self.mean = nn.Linear(512, self.z_dim)
        self.log_std = nn.Linear(512, self.z_dim)

    def forward(self, inputs):
        h = self.main(inputs)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ndf = config.model.ndf
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset

        self.fc = nn.Linear(self.z_dim, self.ndf * 8 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 8, self.ndf * 4, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ndf * 4, self.ndf * 2, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ndf * 2, self.ndf, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ndf, self.channels, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, self.channels, self.image_size, self.image_size) * -0.602,
                                    requires_grad=False)

    def forward(self, inputs):
        h = self.fc(inputs)
        h = h.view(-1, self.ndf * 8, 4, 4)
        h = self.main(h)
        mean = h.view(-1, self.channels, self.image_size, self.image_size)
        return mean, self.log_std


class Score(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.zfc = nn.Sequential(
            nn.Linear(self.z_dim, self.image_size ** 2),
            nn.Softplus()
        )
        self.main = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.nef * 1, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
            nn.Softplus()
        )
        self.flatten = nn.Sequential(
            nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
            nn.Softplus()
        )
        self.score = nn.Linear(512, self.z_dim)

    def forward(self, X, z):
        z = self.zfc(z).view(X.shape[0], 1, X.shape[2], X.shape[3])
        Xz = torch.cat([X, z], dim=1)
        h = self.main(Xz)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        score = self.score(h)
        return score


class ImplicitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.eps_dim = config.model.eps_dim
        self.image_size = config.data.image_size
        self.channels = config.data.channels
        self.dataset = config.data.dataset
        self.noise_fc = nn.Sequential(
            nn.Linear(self.eps_dim, self.image_size ** 2),
            nn.ReLU(inplace=True)
        )
        self.main = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.nef * 1, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2)
        )
        self.flatten = nn.Sequential(
            nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, self.z_dim)

    def forward(self, inputs):
        noise = torch.randn(inputs.shape[0], self.eps_dim, device=inputs.device)
        noise = self.noise_fc(noise).view(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
        noisy_inputs = torch.cat([inputs, noise], dim=1)
        h = self.main(noisy_inputs)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        h = self.fc(h)
        return h
