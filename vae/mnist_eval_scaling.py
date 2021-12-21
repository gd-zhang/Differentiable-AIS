import argparse
import math
import os
import time
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

import sys
sys.path.append("./")
from src.utils import mkdir, get_logger, stochMNIST
from src.sampling.leapfrogs import leapfrogs_and_bounds, leapfrogs_and_bounds_optlr
from src.sampling.ais import annealed_importance_sampling
from src.sampling.hais import hamiltonian_ais

parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--hdim', type=int, default=200,
                    help='number of hidden units (default: 200)')
parser.add_argument('--zdim', type=int, default=50,
                    help='dimension of latent variables (default: 50)')
parser.add_argument('--lf_lrate', type=float, default=0.01,
                    help='lrate for leapfrog step (default: 0.01)')
parser.add_argument('--scaling_factor', type=float, default=0.25,
                    help='how to scale learning rate')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='momentum decay coefficient in leapfrog (default: 0.9)')
parser.add_argument('--linear_beta', action='store_true', default=False,
                    help='whether to use linear schedule')
parser.add_argument('--n_particles', type=int, default=1,
                    help='number of particles for iwae (default: 1)')
parser.add_argument('--ais', action='store_true',
                    help='whether to use annealed importance sampling')
parser.add_argument('--hais', action='store_true',
                    help='whether to use hamiltonian annealed importance sampling')
parser.add_argument('--init_prior', action='store_true',
                    help='whether to ignore the encoder')
parser.add_argument('--gaussian', action='store_true', default=False,
                    help='whether to use Gaussian observation model')
parser.add_argument('--obs_var', type=float, default=0.01,
                    help='observartion variance for Gaussian model.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--seed', type=int, default=2019)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)

kwargs = {'num_workers': 2, 'pin_memory': True, 'worker_init_fn': _init_fn} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    stochMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


def init_logger():
    save_dir = args.resume[:-13]
    print(save_dir)
    if not os.path.isdir(save_dir):
        raise NotFoundError

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_main = os.path.join(path, 'mnist_eval_scaling.py')
    name = 'scaling_c%.2f_lr%.4f_iwae%d' % (args.scaling_factor, 
        args.lf_lrate, args.n_particles)
    if args.ais:
        name += '_ais'
    elif args.hais:
        name += '_hais'
    else:
        name += '_clr'
    if args.linear_beta:
        name += '_lin'

    if args.seed != 2019:
        name += '_seed%d' % args.seed

    logger = get_logger(name, logpath=save_dir+'/', filepath=path_main)
    logger.info(args)
    return save_dir, logger


def log_mean_exp(x):
    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


class Network(nn.Module):
    def __init__(self, latent_dim, hidden_units):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc31 = nn.Linear(hidden_units, latent_dim)
        self.fc32 = nn.Linear(hidden_units, latent_dim)
        self.fc4 = nn.Linear(latent_dim, hidden_units)
        self.fc5 = nn.Linear(hidden_units, hidden_units)
        self.fc6 = nn.Linear(hidden_units, 784)

    def encode(self, x):
        h1 = F.tanh(self.fc2(F.tanh(self.fc1(x))))
        # h1 = F.tanh(self.fc1(x))
        return self.fc31(h1), self.fc32(h1)

    def decode(self, z):
        h3 = F.tanh(self.fc5(F.tanh(self.fc4(z))))
        return self.fc6(h3)


class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_units):
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self._hidden_units = hidden_units

        # self.fc1 = nn.Linear(784, hidden_units)
        # self.fc2 = nn.Linear(hidden_units, hidden_units)
        # self.fc31 = nn.Linear(hidden_units, latent_dim)
        # self.fc32 = nn.Linear(hidden_units, latent_dim)
        # self.fc4 = nn.Linear(latent_dim, hidden_units)
        # self.fc5 = nn.Linear(hidden_units, hidden_units)
        # self.fc6 = nn.Linear(hidden_units, 784)

        self.net = Network(latent_dim, hidden_units)

    # def encode(self, x):
    #     h1 = F.tanh(self.fc2(F.tanh(self.fc1(x))))
    #     return self.fc31(h1), self.fc32(h1)

    # def decode(self, z):
    #     h3 = F.tanh(self.fc5(F.tanh(self.fc4(z))))
    #     return self.fc6(h3)

    def encode(self, x):
        return self.net.encode(x)

    def decode(self, z):
        return self.net.decode(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def tighter_elbo(self, x, n_steps, step_size=0.05, partial=True, 
        gamma=0.9, k=1, init_prior=False, ais=False, hais=False):
        if init_prior:
            mu = torch.zeros(x.shape[0], self._latent_dim, device=x.device, requires_grad=True)
            logvar = torch.zeros(x.shape[0], self._latent_dim, device=x.device, requires_grad=True)
        else:
            mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar, x = mu.repeat(k, 1), logvar.repeat(k, 1), x.repeat(k, 1, 1, 1)
        z = self.reparameterize(mu, logvar)

        def log_likelihood(zz, block_grad=False):
            recon_x_logits = self.decode(zz)
            if args.gaussian:
                log_prob = torch.sum(
                    -0.5 * F.mse_loss(torch.sigmoid(recon_x_logits), x.view(-1, 784), reduction='none') / args.obs_var 
                        - 0.5 * math.log(2 * math.pi) - 0.5 * math.log(args.obs_var), 1)
            else:
                log_prob = -torch.sum(F.binary_cross_entropy_with_logits(recon_x_logits, x.view(-1, 784), reduction='none'), 1)
            log_prob += - 0.5 * (self._latent_dim * math.log(2 * math.pi) + torch.sum(zz ** 2, 1))
            return log_prob

        def log_q(zz):
            diff = zz - mu
            log_q = - 0.5 * (self._latent_dim * math.log(2 * math.pi) + torch.sum(logvar, 1)
                + torch.sum(diff ** 2 / torch.exp(logvar), 1))
            return log_q

        # lrates = step_size * np.linspace(1.0, decay_factor, n_steps)
        assert not (ais and hais)
        betas = None
        if args.linear_beta:
            betas = np.linspace(0, 1, num=n_steps+1)
            
        if ais:
            elbo, z = annealed_importance_sampling(z, log_likelihood, log_q, n_steps, step_size, betas=betas)
        elif hais:
            elbo, z = hamiltonian_ais(z, log_likelihood, log_q, n_steps, step_size, partial, gamma=gamma, betas=betas)
        else:
            lrates = None
            # lrates = step_size * np.linspace(1.0, args.decay_factor, n_steps)
            elbo, z = leapfrogs_and_bounds(z, log_likelihood, log_q, n_steps, step_size, partial, 
                gamma=gamma, lrates=lrates, betas=betas, is_train=False)
        # elbo = log_mean_exp(elbo.view(k, -1).transpose(0, 1))
        elbo = elbo.view(k, -1).transpose(0, 1).mean(1)
        return elbo.sum()


def eval(model, logger, stepsize, iteration):
    model.eval()
    test_loss = 0
    end = time.time()
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        elbo = model.tighter_elbo(data, n_steps=iteration, step_size=stepsize, 
            gamma=args.gamma, k=args.n_particles, 
            init_prior=args.init_prior, ais=args.ais, hais=args.hais)
        test_loss -= elbo.item()

        batch_time = time.time() - end
        end = time.time()
        # logger.info('[{}/{}] Batch: {:.3f}s \tELBO: {:.6f}'.format(batch_idx * len(data),
        #     len(test_loader.dataset), batch_time, elbo.item() / len(data)))


    test_loss /= len(test_loader.dataset)
    logger.info('====> Test set loss: {:.4f}\n'.format(test_loss))


def main():
    save_dir, logger = init_logger()
    model = VAE(args.zdim, args.hdim).to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Getting init model from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model_dict = model.state_dict()
        checkpoint = checkpoint['state_dict']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(checkpoint)

    if args.ais:
        iterations = np.logspace(0, 4, num=9).tolist()
        for iteration in iterations:
            logger.info('Step-size: {:.4f} | Iteration: {}'.format(args.lf_lrate, int(iteration)))
            eval(model, logger, args.lf_lrate, int(iteration))
    else:
        c = args.scaling_factor
        iterations = np.logspace(1, 5, num=9).tolist()
        for iteration in iterations:
            logger.info('Step-size: {:.4f} | Iteration: {}'.format(args.lf_lrate / ((iteration / 10) ** c), int(iteration)))
            eval(model, logger, args.lf_lrate / ((iteration / 10) ** c), int(iteration))

if __name__ == "__main__":
    main()