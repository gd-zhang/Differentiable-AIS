import math
import torch
import numpy as np
from torch import autograd
from src.dist import Normal
from tqdm import tqdm


def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))


def annealed_importance_sampling(s, log_likelihood, log_q, n_steps, 
	step_size, leapfrog_step=10, mass_matrix=None, betas=None):
	
	n_particles = s.shape[0]
	dim = s.shape[1]

	# use different step size for different examples
	step_size = step_size * torch.ones(n_particles, 1, device=s.device)

	if betas is None:
		# betas = torch.linspace(0.0, 1.0, n_steps+1, device=s.device)
		betas = get_schedule(n_steps+1)

	if mass_matrix is None:
		mass_matrix = torch.eye(dim, device=s.device)

	pi = Normal(
		torch.zeros(dim, 1, device=s.device),
		mass_matrix
	)
	inverse_mass_matrix = torch.inverse(mass_matrix)
	kinetic_fn = lambda mom: 0.5 * torch.sum(mom.mm(inverse_mass_matrix) * mom, 1)

	def log_annealed_prob(beta, s):
		return (1 - beta) * log_q(s) + beta * log_likelihood(s)

	def grad_log_annealed_prob(beta, s):
		''' for evaluation, we don't have to create graph '''
		return autograd.grad(log_annealed_prob(beta, s).sum(), s)[0]

	# init log weights
	logw = torch.zeros(n_particles, device=s.device, requires_grad=False)

	# sample momentum variable
	v = pi.sample(n_particles)

	# setup average accept rate
	avg_accept_rate = 0.65 * torch.ones(n_particles, 1, device=s.device)

	for k in range(1, n_steps+1):

		potential_fn = lambda state: -log_annealed_prob(betas[k], state)
		grad_fn = lambda state: grad_log_annealed_prob(betas[k], state)

		# update log weights
		logw += log_annealed_prob(betas[k], s).detach() - log_annealed_prob(betas[k-1], s).detach()
		
		# hmc step with # of leapfrog steps
		new_s, new_v = hmc_step(s, v, step_size, leapfrog_step, grad_fn, inverse_mass_matrix)
		
		# accept & reject with MH rule
		s, v, step_size, avg_accept_rate = hmc_accept_reject(new_s, new_v, 
			s, v, potential_fn, kinetic_fn, step_size, avg_accept_rate)

		# resample momentum variable
		v = pi.sample(n_particles)
	
	return logw, s.detach()


def hmc_step(s, v, step_size, leapfrog_step, grad_fn, inverse_mass_matrix):
	assert leapfrog_step >= 1

	# first do a half step for momentum variable
	v = v + step_size / 2 * grad_fn(s)
	for i in range(1, leapfrog_step+1):
		
		# leapfrog
		s = s + step_size * v.mm(inverse_mass_matrix)
		
		if i < leapfrog_step:
			v = v + step_size * grad_fn(s)
		else:
			v = v + step_size / 2 * grad_fn(s)
	v = -v

	return s.detach(), v.detach()


def hmc_accept_reject(new_s, new_v, old_s, old_v, potential_fn, kinetic_fn, step_size, avg_accept_rate):
	current_energy = potential_fn(new_s) + kinetic_fn(new_v)
	previous_energy = potential_fn(old_s) + kinetic_fn(old_v)

	prob = torch.clamp_max(torch.exp(previous_energy - current_energy), 1.)

	with torch.no_grad():
		uniform_sample = torch.rand(prob.size(), device=new_s.device)
		accept = (prob > uniform_sample).float()
		s = new_s.mul(accept.view(-1, 1)) + old_s.mul(1. - accept.view(-1, 1))
		v = new_v.mul(accept.view(-1, 1)) + old_v.mul(1. - accept.view(-1, 1))

		# exponential moving average with beta = 0.9
		avg_accept_rate = 0.9 * avg_accept_rate + 0.1 * accept.view(-1, 1)

		# target accept rate = 0.65
		criteria = (avg_accept_rate > 0.65).float()

		# adapt step size according to avg_accept_rate
		adapt = 1.02 * criteria + 0.98 * (1. - criteria)
		step_size = step_size.mul(adapt).clamp(1e-4, .5)
	# print('step-size: %.4f' % step_size.mean().cpu().numpy())
	# print('accept-rate: %.4f' % accept.mean().cpu().numpy())
	s.requires_grad_()

	return s, v, step_size, avg_accept_rate