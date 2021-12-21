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


def leapfrogs_and_bounds(s, log_likelihood, log_q, n_steps,
        step_size, partial=False, gamma=0.9, mass_matrix=None, 
        lrates=None, betas=None, block_grad=False, is_train=True):
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

    # s.requires_grad = True

    def log_annealed_prob(beta, s):
        return (1 - beta) * log_q(s) + beta * log_likelihood(s, block_grad)

    def grad_log_annealed_prob(beta, s):
        ''' it's important to set create_graph=True '''
        with torch.enable_grad():
            s.requires_grad_()
            grad = autograd.grad(log_annealed_prob(beta, s).sum(), s, create_graph=is_train)[0]
        return grad

    # sample initial momentum
    v = pi.sample(n_particles)

    with torch.set_grad_enabled(is_train):

        elbo = - log_q(s)
        for k in range(1, n_steps+1):
            elbo = elbo - pi.log_prob(v)

            # leapfrog
            s = s + lrates[k] / 2 * v.mm(inverse_mass_matrix)
            v = v + lrates[k] * grad_log_annealed_prob(betas[k], s)
            s = s + lrates[k] / 2 * v.mm(inverse_mass_matrix)

            elbo = elbo + pi.log_prob(v)

            if partial:
                # partial_refreshment
                v = gamma * v + math.sqrt(1 - math.pow(gamma, 2)) * pi.sample(n_particles)
            else:
                v = pi.sample(n_particles)

        elbo = elbo + log_likelihood(s)

    return elbo, s


def leapfrogs_and_bounds_optlr(s, log_likelihood, log_q, n_steps,
        step_size, partial=False, gamma=0.9, mass_matrix=None):
    
    init_log_lrates = math.log(step_size) * np.ones(n_steps)
    bounds = scipy.optimize.Bounds(-np.infty, -1.0)
    # log_lrates = torch.tensor(llrates_np, dtype=s.dtype, device=s.device, requires_grad=True)

    def func_fn(log_lrates):
        log_lrates = torch.tensor(log_lrates, dtype=s.dtype, device=s.device, requires_grad=True)
        elbo, _ = leapfrogs_and_bounds(s, log_likelihood, log_q, n_steps, 
            step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))
        return -elbo.sum().data.cpu().numpy()

    def grad_fn(log_lrates):
        log_lrates = torch.tensor(log_lrates, dtype=s.dtype, device=s.device, requires_grad=True)
        elbo, _ = leapfrogs_and_bounds(s, log_likelihood, log_q, n_steps, 
            step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))
        loss = -elbo.sum()
        return autograd.grad(loss, log_lrates)[0].data.cpu().numpy().astype(np.float64)

    res = scipy.optimize.minimize(func_fn, init_log_lrates, jac=grad_fn, bounds=bounds)
    log_lrates = torch.tensor(res.x, dtype=s.dtype, device=s.device)

    return leapfrogs_and_bounds(s, log_likelihood, log_q, n_steps, 
        step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))





