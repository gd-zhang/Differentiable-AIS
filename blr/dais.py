import jax
import jax.numpy as np
import jax.random as random
from jax.scipy.stats import multivariate_normal
from jax.config import config 
config.update("jax_enable_x64", True)
from functools import partial
from matplotlib import pyplot as plt

from tqdm import tqdm
import numpy as onp
import math
import time
import argparse
import scipy.optimize


parser = argparse.ArgumentParser(description='Bayesian Linear Regression')
parser.add_argument('--gamma', type=float, default=0.0, help='momentum refreshment')
parser.add_argument('--num-points', type=int, default=10000, help='dataset size')
parser.add_argument('--bsize', type=int, default=10000, help='batch size')
parser.add_argument('--dim', type=int, default=10, help='dimension of input')
parser.add_argument('--obs-var', type=float, default=1.0, help='observation variance')
args = parser.parse_args()


colors = ["r", "g", "b", "c"]

def init_plotting():
  plt.rcParams['figure.figsize'] = (10, 6)
  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams['font.size'] = 16
  plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']
  plt.rcParams["legend.framealpha"] = 0.7
  plt.rcParams['lines.linewidth'] = 2.5
  plt.rcParams['lines.markersize'] = 10

def negative_entropy(cov):
  "H(p) = 0.5 * ln|Sigma| + D/2 * (1 + ln(2pi))"
  d = np.shape(cov)[0]
  return -0.5 * d * np.log(2 * np.pi) - 0.5 * d - 0.5 * np.linalg.slogdet(cov)[1]

def expect_log_p(q_mu, q_cov, p_mu, p_cov):
  "Computing the expectation of log_p under distribution q"
  d = np.shape(q_cov)[0]
  p_prec = np.linalg.inv(p_cov)
  log_p = -0.5 * d * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(p_cov)[1] \
    - 0.5 * (q_mu - p_mu).T @ p_prec @ (q_mu - p_mu) - 0.5 * np.trace(p_prec @ q_cov)
  return log_p[0, 0]

def block_matrix(A, B, C, D):
  "autograd DOESN'T support np.block"
  return np.concatenate((np.concatenate((A, B), axis=1), np.concatenate((C, D), axis=1)), axis=0)

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def repeat(x, piece, length):
  interval = int(length / piece)
  x = np.tile(x, (interval, 1)).T.flatten()
  x = np.concatenate((x, x[-1] * np.ones((length - interval * piece))), axis=0)
  return x

def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))


class Sampler:
  def __init__(self, n_fea, n_data, obs, obs_var, cov_data, w_star, w_prior, cov_prior):
    self._n_fea = n_fea
    self._n_data = n_data
    self._obs = obs
    self._obs_var = obs_var

    # important: data covaraince is precision matrix
    self._prec_lld = cov_data / obs_var
    self._cov_lld = np.linalg.inv(self._prec_lld)
    self._w_star = w_star
    self._w_prior = w_prior
    self._cov_prior = cov_prior
    self._prec_prior = np.linalg.inv(cov_prior)

  def expect_lld(self, q_mu, q_cov):
    "Compute the expectation of log-likelihood under distribution q"
    lld = -0.5 * self._n_data * np.log(2 * np.pi) - 0.5 * self._n_data * np.log(self._obs_var) \
      - 0.5 * (q_mu - self._w_star).T @ self._prec_lld @ (q_mu - self._w_star) - 0.5 * np.trace(self._prec_lld @ q_cov) \
      - 0.5 * np.sum(self._obs ** 2) / self._obs_var + 0.5 * self._w_star.T @ self._prec_lld @ self._w_star
    return lld[0, 0]

  def expect_log_pi(self, q_mu, q_cov):
    # assume pi is N(0, I)
    d = np.shape(q_cov)[0]
    log_p = -0.5 * d * np.log(2 * np.pi) \
      - 0.5 * q_mu.T @ q_mu - 0.5 * np.trace(q_cov)
    return log_p[0, 0]

  def log_marginal_likelihood(self):
    cov_pos = np.linalg.inv(self._prec_lld + self._prec_prior)
    prec_pos = self._prec_lld + self._prec_prior
    w_pos = cov_pos @ (self._prec_lld @ self._w_star + self._prec_prior @ self._w_prior)
    e_pos, _ = np.linalg.eigh(cov_pos)
    e_prior, _ = np.linalg.eigh(self._cov_prior)

    log_ml = 0.5 * np.sum(np.log(e_pos)) - 0.5 * np.sum(np.log(e_prior)) - 0.5 * self._n_data * (np.log(2 * np.pi) + np.log(self._obs_var)) \
      + 0.5 * w_pos.T @ prec_pos @ w_pos - 0.5 * self._w_prior.T @ self._prec_prior @ self._w_prior \
      - 0.5 * np.sum(self._obs ** 2) / self._obs_var
    return log_ml[0, 0]

  def lower_bound(self, w, w_cov, log_ratio, cov_init=None):
    if cov_init is None:
      cov_init = self._cov_prior
    return self.expect_lld(w, w_cov) + expect_log_p(w, w_cov, self._w_prior, self._cov_prior) \
      - negative_entropy(cov_init) + log_ratio

  @partial(jax.jit, static_argnums=(0,))
  def transition(self, w, v, covar, log_ratio, lrate, beta, gamma, grad_noise):
    # get precision matrix of posterior for current beta
    prec_pos = self._prec_prior + beta * self._prec_lld
    # the following w_pos is not exactly posterior mean, it's actually prec_pos @ mean_pos
    w_pos = self._prec_prior @ self._w_prior + beta * self._prec_lld @ self._w_star

    # transition matrices
    Tww = np.eye(self._n_fea) - 0.5 * lrate ** 2 * prec_pos
    Twv = lrate * np.eye(self._n_fea) - 0.25 * lrate ** 3 * prec_pos
    Tvv = np.eye(self._n_fea) - 0.5 * lrate ** 2 * prec_pos
    Tvw = - lrate * prec_pos
    T = block_matrix(Tww, Twv, Tvw, Tvv)

    # covariance matrix of gradient
    grad_cov = beta * self._prec_lld

    # apply transition 
    w_new = Tww @ w + Twv @ v + 0.5 * lrate ** 2 * w_pos
    v_new = Tvw @ w + Tvv @ v + lrate * w_pos
    covar_new = T @ covar @ T.T 

    # add gradient noise
    covar_new += grad_noise * block_matrix(0.25 * lrate ** 4 * grad_cov, 0.5 * lrate ** 3 * grad_cov, 
                                             0.5 * lrate ** 3 * grad_cov, lrate ** 2 * grad_cov)
    

    # compute log density ratio term
    v_cov_new = covar_new[self._n_fea:, self._n_fea:]
    v_cov = covar[self._n_fea:, self._n_fea:]
    log_ratio += self.expect_log_pi(v_new, v_cov_new) - self.expect_log_pi(v, v_cov)
    # log_ratio += expect_log_p(v_new, v_cov_new, np.zeros((self._n_fea, 1)), np.eye(self._n_fea)) \
    #   - expect_log_p(v, v_cov, np.zeros((self._n_fea, 1)), np.eye(self._n_fea))

    # reset with full/partial momentum refreshment
    w = w_new

    v = gamma * v_new
    covar = block_matrix(covar_new[:self._n_fea, :self._n_fea], 
                         covar_new[:self._n_fea, self._n_fea:] * gamma, 
                         covar_new[self._n_fea:, :self._n_fea] * gamma,
                         covar_new[self._n_fea:, self._n_fea:] * gamma ** 2 + (1 - gamma ** 2) * np.eye(self._n_fea))

    return w, v, covar, log_ratio

  # @partial(jax.jit, static_argnums=(0,))
  def simulate_ksteps(self, lrates, betas, steps, gamma=0.0, grad_noise=0.0, w_init=None, cov_init=None):
    '''
    lrates: step sizes for every iteration -> shape (T, )
    betas: annealing coefficient for every iteration -> shape (T, )
    steps: sampling steps T
    gamma: momentum refreshment coefficient -> similar to momentum in optimization
    grad_noise: noise level -> 1/B
    w_init, cov_init: pre-defined initial distribution, otherwise starts with prior distribution
    '''

    # if w_init is not specified, then start with prior distribution
    if w_init is None:
      w_init = self._w_prior
    if cov_init is None:
      cov_init = self._cov_prior

    # assert len(lrates) == len(betas) == steps

    w, w_cov = w_init, cov_init
    v, v_cov = np.zeros((self._n_fea, 1)), np.eye(self._n_fea)
    covar = block_matrix(w_cov, np.zeros_like(w_cov), np.zeros_like(v_cov), v_cov)

    log_ratio = 0
    w_list = []
    log_ratio_list = []
    for i, (lrate, beta) in tqdm(enumerate(zip(lrates, betas))):
      w, v, covar, log_ratio = self.transition(w, v, covar, log_ratio, lrate, beta, gamma, grad_noise)

    # return w_list, log_ratio_list
    return (w, covar[:self._n_fea, :self._n_fea]), log_ratio


# PARTICLE SAMPLER CLASS 
class ParticleSampler:
  def __init__(self, inputs, obs, obs_var, w_prior, cov_prior):
    self._n_fea = inputs.shape[1]
    self._n_data = inputs.shape[0]

    self._inputs = inputs
    self._obs = obs
    self._obs_var = obs_var

    self._w_prior = w_prior
    self._cov_prior = cov_prior
    self._prec_prior = np.linalg.inv(cov_prior)

    self._w_momentum = np.zeros((self._n_fea, 1))
    self._cov_momentum = np.eye(self._n_fea)
        
  def sample_normal(self, key, mean, cov, particles):
    return random.multivariate_normal(key, mean, cov, (particles, ))

  def sample_prior(self, key, particles):
    return self.sample_normal(key, self._w_prior.squeeze(-1), self._cov_prior, particles)

  def sample_momentum(self, key, particles):
    return self.sample_normal(key, self._w_momentum.squeeze(-1), self._cov_momentum, particles)

  def log_p_prior(self, w):
    # return self.log_p_normal(self._w_prior.squeeze(-1), self._cov_prior, w) 
    return multivariate_normal.logpdf(w, self._w_prior.squeeze(-1), self._cov_prior)

  def log_p_pi(self, v):
    # return self.log_p_normal(self._w_momentum.squeeze(-1), self._cov_momentum, v)
    return multivariate_normal.logpdf(v, self._w_momentum.squeeze(-1), self._cov_momentum)

  def log_p_likelihood(self, inputs, obs, w):
    n = len(obs)
    result = (- 0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(self._obs_var)
              - 0.5 * np.sum(w @ (inputs.T @ inputs / self._obs_var) * w, 1)
              + np.squeeze(w @ (inputs.T @ obs / self._obs_var))
              - 0.5 * np.sum(self._obs ** 2) / self._obs_var
              )
    return result * self._n_data / n

  @partial(jax.jit, static_argnums=(0,7,))
  def transition(self, key, w, v, lrate, beta, gamma, bsize):
    key, subkey1, subkey2 = random.split(key, 3)
    indices = random.permutation(subkey1, self._n_data)[:bsize]
    inputs = self._inputs[indices]
    obs = self._obs[indices]

    log_ratio = - self.log_p_pi(v)
    w = w + lrate / 2 * v
    w_grad = (- (inputs.T @ inputs / self._obs_var * beta * self._n_data / bsize
                 + self._prec_prior) @ w[..., np.newaxis]
              + (inputs.T @ obs / self._obs_var * beta * self._n_data / bsize
                 + self._prec_prior @ self._w_prior)).squeeze(-1)
    v_star = v + lrate * w_grad
    w = w + lrate / 2 * v_star
    v = gamma * v_star + (1 - gamma ** 2) ** 0.5 * self.sample_momentum(subkey2, len(v))
    log_ratio += self.log_p_pi(v_star)

    return w, v, log_ratio, key

  def simulate_ksteps(self, key, lrates, betas, steps, particles, gamma=None, bsize=None):
    '''
    lrates: step size for every iteration -> shape (T, )
    betas: annealing coefficient for every iteration -> shape (T, )
    particles: number of particles
    steps: sampling steps T
    gamma: momentum refreshment coefficient -> similar to momentum in optimization
    bsize: batch size for every iteration -> shape (T, ) full batch if None
    '''

    # assert len(lrates) == len(betas) == steps
    if bsize is None:
      bsize = self._n_data

    key, subkey1, subkey2 = random.split(key, 3)
    w = self.sample_prior(subkey1, particles)
    v = self.sample_momentum(subkey2, particles)
    w_init = w
    log_ratio_sum = 0.0
    for i, (lrate, beta) in tqdm(enumerate(zip(lrates, betas))):
      w, v, log_ratio, key = self.transition(key, w, v, lrate, beta, gamma, bsize)
      log_ratio_sum += log_ratio

    return w, w_init, log_ratio_sum

  def lower_bound(self, w, w_init, log_ratio):
    inputs = self._inputs
    obs = self._obs
    return (self.log_p_likelihood(inputs, obs, w) + 
      self.log_p_prior(w) - self.log_p_prior(w_init) + log_ratio)



def main():
  # parameters for data generation
  key = random.PRNGKey(2021)
  d = args.dim
  n = args.num_points
  obs_var = args.obs_var
  key, subkey = random.split(key, 2)
  inputs = random.normal(subkey, (n, d)) * 0.1
  cov_data = inputs.T @ inputs

  key, subkey1, subkey2 = random.split(key, 3)
  obs = inputs @ random.normal(subkey1, (d, 1)) + (obs_var ** 0.5) * random.normal(subkey2, (n, 1))
  w_star = np.linalg.inv(cov_data) @ inputs.T @ obs
  w_prior = np.zeros((d, 1))
  cov_prior = np.eye(d)

  cov_pos = np.linalg.inv(cov_data / obs_var + np.linalg.inv(cov_prior))
  w_pos = cov_pos @ ((cov_data / obs_var) @ w_star + np.linalg.inv(cov_prior) @ w_prior)
  print("\n==> printing true posterior mean and covariance:")
  print(w_pos[:, 0])
  print(cov_pos)

  # instantiate the sampler object
  sampler = Sampler(d, n, obs, obs_var, cov_data, w_star, w_prior, cov_prior)
  
  # need to feed in raw input and observations
  particle_sampler = ParticleSampler(inputs, obs, obs_var, w_prior, cov_prior)

  log_ml = sampler.log_marginal_likelihood()
  print("\n==> log marginal likelihood: %.4f" % log_ml)

  e, _ = onp.linalg.eig(cov_data / obs_var)
  L = np.max(e)

  iterations = np.logspace(1, 7, num=13)

  plt.figure()
  init_plotting()
  for i, c in enumerate([1.0/4, 1.0/3, 1.0/2]):
    gap_list, gap_sample_list, gap_sample_iwae_list = [], [], []
    for iteration in iterations:
      betas = np.linspace(0, 1.0, int(iteration))
      lrates = (1 / (1 + betas * L)) ** 0.5 / ((iteration / 10) ** c)

      if args.bsize == args.num_points:
        w_list, log_ratio = sampler.simulate_ksteps(lrates, betas, int(iteration), gamma=args.gamma)
        lb = sampler.lower_bound(w_list[0], w_list[1], log_ratio)
        gap = log_ml - lb
        gap_list.append(gap)
      
      w, w_init, log_ratio = particle_sampler.simulate_ksteps(key, lrates, betas, 
        int(iteration), 100, gamma=args.gamma, bsize=args.bsize)
      lb = particle_sampler.lower_bound(w, w_init, log_ratio)
      gap = log_ml - lb
      gap_sample_list.append(gap)

    plt.plot(iterations, np.array(gap_sample_list).mean(1), color=colors[i], label='c = %.3f' % c)
    if args.bsize == args.num_points:
      plt.plot(iterations, gap_list, color=colors[i], linestyle='dotted')
      plt.plot(iterations, gap_list[-1] * np.logspace(-6, 0, num=13) ** (2*c -1), color=colors[i], linestyle='--')

  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Num of Intermediate Distributions')
  plt.ylabel('Suboptimality Gap')
  plt.tight_layout()
  plt.grid(linestyle='--', linewidth=1.5)
  plt.legend()
  plt.savefig('scaling_gamma%.2f.pdf' % args.gamma)
  plt.close()


if __name__ == "__main__":
    main()