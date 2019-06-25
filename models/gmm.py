import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

class Gaussian4SVI(nn.Module):
    def __init__(self, batch_size, dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(batch_size, dim))
        self.mean = nn.Parameter(torch.zeros(batch_size, dim))

    def forward(self, X):
        return self.mean, self.log_std
