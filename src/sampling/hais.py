import math
import torch
import numpy as np
from torch import autograd, optim
from src.dist import Normal
import scipy.optimize
from tqdm import tqdm


def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))


def hamiltonian_ais(s, log_likelihood, log_q, n_steps,
        step_size, partial=False, gamma=0.9, mass_matrix=None, 
        lrates=None, betas=None):
    """
    s: partcle state: n_particles x d
    """
    
    n_particles = s.shape[0]
    dim = s.shape[1]

    if n_steps > 0:
        if lrates is None:
            lrates = step_size * torch.ones(n_steps+1, device=s.device)
        if betas is None:
            # betas = torch.linspace(1.0/n_steps, 1.0, n_steps, device=s.device)
            betas = get_schedule(n_steps+1)
    else:
        return - log_q(s) + log_likelihood(s), s

    if mass_matrix is None:
        mass_matrix = torch.eye(dim, device=s.device)

    pi = Normal(
        torch.zeros(dim, 1, device=s.device),
        mass_matrix
    )
    inverse_mass_matrix = torch.inverse(mass_matrix)
    kinetic_fn = lambda mom: 0.5 * torch.sum(mom.mm(inverse_mass_matrix) * mom, 1)

    # s.requires_grad = True

    def log_annealed_prob(beta, s):
        return (1 - beta) * log_q(s) + beta * log_likelihood(s)

    def grad_log_annealed_prob(beta, s):
        ''' it's important to set create_graph=True '''
        with torch.enable_grad():
            s.requires_grad_()
            grad = autograd.grad(log_annealed_prob(beta, s).sum(), s)[0]
        return grad

    # init log weights
    logw = torch.zeros(n_particles, device=s.device, requires_grad=False)

    # sample initial momentum
    v = pi.sample(n_particles)

    with torch.no_grad():

        for k in range(1, n_steps+1):

            potential_fn = lambda state: -log_annealed_prob(betas[k], state)
            grad_fn = lambda state: grad_log_annealed_prob(betas[k], state)

            # update log weights
            logw += log_annealed_prob(betas[k], s).detach() - log_annealed_prob(betas[k-1], s).detach()

            # leapfrog
            new_s = s + lrates[k] / 2 * v.mm(inverse_mass_matrix)
            new_v = v + lrates[k] * grad_log_annealed_prob(betas[k], new_s)
            new_s = new_s + lrates[k] / 2 * new_v.mm(inverse_mass_matrix)

            # accept & reject with MH rule
            s, v = accept_reject(new_s, new_v, s, -v, potential_fn, kinetic_fn)

            if partial:
                # partial_refreshment
                v = gamma * v + math.sqrt(1 - math.pow(gamma, 2)) * pi.sample(n_particles)
            else:
                v = pi.sample(n_particles)

    return logw, s.detach()


def accept_reject(new_s, new_v, old_s, old_v, potential_fn, kinetic_fn):
    current_energy = potential_fn(new_s) + kinetic_fn(new_v)
    previous_energy = potential_fn(old_s) + kinetic_fn(old_v)

    prob = torch.clamp_max(torch.exp(previous_energy - current_energy), 1.)

    with torch.no_grad():
        uniform_sample = torch.rand(prob.size(), device=new_s.device)
        accept = (prob > uniform_sample).float()
        s = new_s.mul(accept.view(-1, 1)) + old_s.mul(1. - accept.view(-1, 1))
        v = new_v.mul(accept.view(-1, 1)) + old_v.mul(1. - accept.view(-1, 1))

    s.requires_grad_()

    return s, v





