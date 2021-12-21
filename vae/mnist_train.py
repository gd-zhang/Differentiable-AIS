import argparse
import math
import os
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
from src.sampling.leapfrogs import leapfrogs_and_bounds
from ckpt_utils.save_func import checkpoint_save
from ckpt_utils.load_func import latest_checkpoint_load

parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--lrate', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of epochs to train (default: 1500)')
# parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800, 900], 
#                     help='Decrease learning rate at these epochs.')
parser.add_argument('--hdim', type=int, default=200,
                    help='number of hidden units (default: 200)')
parser.add_argument('--zdim', type=int, default=50,
                    help='dimension of latent variables (default: 20)')
parser.add_argument('--lf_step', type=int, default=0,
                    help='number of leapfrog step (default: 0)')
parser.add_argument('--lf_lrate', type=float, default=0.01,
                    help='lrate for leapfrog step (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='momentum decay coefficient in leapfrog (default: 0.9)')
parser.add_argument('--n_particles', type=int, default=1,
                    help='number of particles for iwae (default: 1)')
parser.add_argument('--l2_logit', type=float, default=0.01,
                    help='l2 regularization for beta logits (default: 0.01)')
parser.add_argument('--adapt_beta', action='store_true', default=False,
                    help='whether to adapt annealing scheme')
parser.add_argument('--gaussian', action='store_true', default=False,
                    help='whether to use Gaussian observation model')
parser.add_argument('--block_grad', action='store_true', default=False,
                    help='whether to block grad for decoder in sampling')
parser.add_argument('--obs_var', type=float, default=0.01,
                    help='observartion variance for Gaussian model.')
parser.add_argument('--vae', action='store_true', default=False,
                    help='use VAE ELBO even with multiple particles')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--ckpt-dir', type=str, default=None)
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
train_loader = torch.utils.data.DataLoader(
    stochMNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    stochMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


def init_logger():
    base_dir = os.path.join('results', 'vae', 'mnist')
    folder_name = 'bs%d_lr%.4f_hdim%d_zdim%d' % (args.batch_size, 
        args.lrate, args.hdim, args.zdim)
    if args.vae:
        folder_name += '_vae%d' % args.n_particles
    else:
        folder_name += '_iwae%d' % args.n_particles

        if args.lf_step > 0:
            folder_name += '_lfs%d_lr%.3f_gamma%.2f' % (args.lf_step, 
                args.lf_lrate, args.gamma)
            if args.adapt_beta:
                folder_name += '_adp_l2l%.4f' % args.l2_logit
            if args.block_grad:
                folder_name += '_blk'
    if args.gaussian:
        folder_name += '_gau_var%.3f' % args.obs_var
    folder_name += '_seed%d' % args.seed
    save_dir = os.path.join(base_dir, folder_name)
    if not os.path.isdir(save_dir):
        mkdir(save_dir+'/imgs')

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_main = os.path.join(path, 'mnist_train.py')
    logger = get_logger('log', logpath=save_dir+'/', filepath=path_main)
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
    def __init__(self, latent_dim, hidden_units, n_steps):
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self._hidden_units = hidden_units

        self.net = Network(latent_dim, hidden_units)
        if args.block_grad:
            self.net_copy = Network(latent_dim, hidden_units)
            for _, p in self.net_copy.named_parameters():
                p.requires_grad = False
            self.sync()

        if n_steps > 0 and args.adapt_beta:
            self.beta_logits = nn.Parameter(torch.zeros(n_steps))
            # self.lrate_logits = nn.Parameter(torch.zeros(n_steps))

    def sync(self):
        assert hasattr(self, 'net_copy')

        # for network with batch norm, have to take into account running mean
        for lb, lc in zip(self.net.modules(), self.net_copy.modules()):
            if isinstance(lb, nn.Conv2d) or isinstance(lb, nn.Linear):
                lc.weight.data = lb.weight.data.clone()
                lc.bias.data = lb.bias.data.clone()

    def encode(self, x, block_grad=False):
        assert (block_grad == False) or hasattr(self, 'net_copy')
        return self.net_copy.encode(x) if block_grad else self.net.encode(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, block_grad=False):
        assert (block_grad == False) or hasattr(self, 'net_copy')
        return self.net_copy.decode(z) if block_grad else self.net.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def elbo(self, x, k=1):
        mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar, x = mu.repeat(k, 1), logvar.repeat(k, 1), x.repeat(k, 1, 1, 1)
        z = self.reparameterize(mu, logvar)
        recon_x_logits = self.decode(z)
        recon_x = torch.sigmoid(recon_x_logits)
        if args.gaussian:
            NLLD = torch.sum(
                0.5 * F.mse_loss(torch.sigmoid(recon_x_logits), x.view(-1, 784), reduction='none') / args.obs_var 
                + 0.5 * math.log(2 * math.pi) + 0.5 * math.log(args.obs_var), 1)
        else:
            NLLD = torch.sum(F.binary_cross_entropy_with_logits(recon_x_logits, x.view(-1, 784), reduction='none'), 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        elbo = - NLLD - KLD
        elbo = elbo.view(k, -1).transpose(0, 1).mean(1)
        return elbo.sum(), recon_x

    def tighter_elbo(self, x, n_steps, step_size=0.05, partial=True, gamma=0.9, k=1, block_grad=False, is_train=True):
        mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar, x = mu.repeat(k, 1), logvar.repeat(k, 1), x.repeat(k, 1, 1, 1)
        z = self.reparameterize(mu, logvar)

        def log_likelihood(z, block_grad=False):
            recon_x_logits = self.decode(z, block_grad=block_grad)
            if args.gaussian:
                log_prob = torch.sum(
                    - 0.5 * F.mse_loss(torch.sigmoid(recon_x_logits), x.view(-1, 784), reduction='none') / args.obs_var 
                    - 0.5 * math.log(2 * math.pi) - 0.5 * math.log(args.obs_var), 1)
            else:
                log_prob = -torch.sum(F.binary_cross_entropy_with_logits(recon_x_logits, x.view(-1, 784), reduction='none'), 1)
            log_prob += - 0.5 * (self._latent_dim * math.log(2 * math.pi) + torch.sum(z ** 2, 1))
            return log_prob

        def log_q(z):
            diff = z - mu
            log_q = - 0.5 * (self._latent_dim * math.log(2 * math.pi) + torch.sum(logvar, 1)
                + torch.sum(diff ** 2 / torch.exp(logvar), 1))
            return log_q

        betas = None
        lrates = None
        if n_steps > 0:
            if args.adapt_beta:
                beta_deltas = F.softmax(self.beta_logits, dim=0)
                betas = torch.cumsum(beta_deltas, dim=0)
                betas = torch.cat((torch.zeros(1, device=device), betas))
            else:
                betas = np.linspace(0, 1, num=n_steps+1)
            # lrates = 0.1 * torch.sigmoid(self.lrate_logits)
            # lrates = step_size / (1 + np.arange(n_steps))
            # lrates = step_size * np.linspace(1.0, 0.1, n_steps)
        elbo, z = leapfrogs_and_bounds(z, log_likelihood, log_q, n_steps, step_size, partial, 
            gamma=gamma, lrates=lrates, betas=betas, block_grad=block_grad, is_train=is_train)
        elbo = log_mean_exp(elbo.view(k, -1).transpose(0, 1))
        recon_x = torch.sigmoid(self.decode(z[:args.batch_size]))
        return elbo.sum(), recon_x


def train(model, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if args.vae:
            elbo, _ = model.elbo(data, k=args.n_particles)
        else:
            elbo, _ = model.tighter_elbo(data, args.lf_step, args.lf_lrate, 
                gamma=args.gamma, k=args.n_particles, block_grad=args.block_grad, is_train=True)
        loss = -elbo
        total_loss = loss
        if args.lf_step > 0 and args.adapt_beta:
            # total_loss += args.l2_logit * torch.sum(model.beta_logits ** 2) * args.batch_size
            beta_deltas = F.softmax(model.beta_logits, dim=0)
            total_loss += args.l2_logit * torch.sum(beta_deltas * torch.log(beta_deltas)) * args.batch_size
        total_loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if args.block_grad:
            model.sync()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(model, optimizer, epoch, logger, save_dir):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        elbo, recon_data = model.tighter_elbo(data, args.lf_step, args.lf_lrate, 
            gamma=args.gamma, k=args.n_particles)
        test_loss -= elbo.item()
        # if i == 0:
        #     n = min(data.size(0), 8)
        #     comparison = torch.cat([data[:n], 
        #         recon_data.view(args.batch_size, 1, 28, 28)[:n]])
        #     save_image(comparison.cpu(), 
        #         save_dir+'/imgs/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    logger.info('====> Test set loss: {:.4f}'.format(test_loss))

def save_checkpoint(state, checkpoint, filename='ckpt.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main():
    save_dir, logger = init_logger()

    model = VAE(args.zdim, args.hdim, args.lf_step).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lrate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    # load checkpoint if it exists
    if args.ckpt_dir is not None:
        checkpoint = latest_checkpoint_load(args.ckpt_dir)
    else:
        checkpoint = None
    if checkpoint != None:
        logger.info('==> restore!')
        model.load_state_dict(checkpoint[0]['state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer'])
        scheduler.load_state_dict(checkpoint[0]['scheduler'])
        start_epoch = checkpoint[0]['epoch']
        logger.info("Loaded a checkpoint\n")
    else:
        start_epoch = 1

    print(model)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, optimizer, epoch, logger)
        test(model, optimizer, epoch, logger, save_dir)
        scheduler.step()

        if args.ckpt_dir is not None:
            checkpoint_name = checkpoint_save({'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1}, args.ckpt_dir)

        if args.lf_step > 0 and args.adapt_beta:
            betas = torch.cumsum(F.softmax(model.beta_logits, dim=0), dim=0)
            logger.info('====> annealing scheme: %s' % np.array2string(betas.data.cpu().numpy()))
        # with torch.no_grad():
        #     sample = torch.randn(64, args.zdim).to(device)
        #     sample = torch.sigmoid(model.decode(sample)).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                save_dir+'/imgs/sample_' + str(epoch) + '.png')

    save_checkpoint({'state_dict': model.state_dict()}, save_dir)

if __name__ == "__main__":
    main()