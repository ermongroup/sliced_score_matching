import torch
import torch.autograd as autograd
import numpy as np

# single_sliced_score_matching and sliced_VR_score_matching implement a basic version of SSM
# with only M=1. These are used in density estimation experiments for DKEF.
def single_sliced_score_matching(energy_net, samples, noise=None, detach=False, noise_type='radermacher'):
    samples.requires_grad_(True)
    if noise is None:
        vectors = torch.randn_like(samples)
        if noise_type == 'radermacher':
            vectors = vectors.sign()
        elif noise_type == 'sphere':
            vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * np.sqrt(vectors.shape[-1])
        elif noise_type == 'gaussian':
            pass
        else:
            raise ValueError("Noise type not implemented")
    else:
        vectors = noise

    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    if detach:
        loss1 = loss1.detach()
    grad2 = autograd.grad(gradv, samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)
    if detach:
        loss2 = loss2.detach()

    loss = (loss1 + loss2).mean()
    return loss, grad1, grad2


def sliced_VR_score_matching(energy_net, samples, noise=None, detach=False, noise_type='radermacher'):
    samples.requires_grad_(True)
    if noise is None:
        vectors = torch.randn_like(samples)
        if noise_type == 'radermacher':
            vectors = vectors.sign()
        elif noise_type == 'gaussian':
            pass
        else:
            raise ValueError("Noise type not implemented")
    else:
        vectors = noise

    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.norm(grad1, dim=-1) ** 2 * 0.5
    if detach:
        loss1 = loss1.detach()
    grad2 = autograd.grad(gradv, samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)
    if detach:
        loss2 = loss2.detach()

    loss = (loss1 + loss2).mean()
    return loss, grad1, grad2


# General implementations of SSM and SSM_VR for arbitrary numbers of particles
def sliced_score_matching(energy_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    logp = -energy_net(dup_samples).sum()
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_matching_vr(energy_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    logp = -energy_net(dup_samples).sum()
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_estimation(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_estimation_vr(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()