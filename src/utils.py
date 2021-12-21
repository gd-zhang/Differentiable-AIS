import os
import errno
import logging
import time

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
import urllib.request
import scipy.io


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


class stochMNIST(datasets.MNIST):
    """ Gets a new stochastic binarization of MNIST at each call. """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = torch.bernoulli(img)  # stochastically binarize
        return img, target

    def get_mean_img(self):
        imgs = self.train_data.type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img


class omniglot(data.Dataset):
    """ omniglot dataset """
    url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

    def __init__(self, root, train=True, transform=None, download=False):
        # we ignore transform.
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if download: self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 28)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        img = torch.bernoulli(img)  # stochastically binarize
        return img, torch.tensor(-1)  # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')

        omni_raw = scipy.io.loadmat(os.path.join(self.root, 'chardata.mat'))
        data_str = 'data' if train else 'testdata'
        data = reshape_data(omni_raw[data_str].T.astype('float32'))
        return data

    def get_mean_img(self):
        return self.data.mean(0)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading from {}...'.format(self.url))
        local_filename = os.path.join(self.root, 'chardata.mat')
        urllib.request.urlretrieve(self.url, local_filename)
        print('Saved to {}'.format(local_filename))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'chardata.mat'))
