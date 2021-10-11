import os
from os.path import join

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image

from tfpnp.data.util import scale_height, scale_width
from tfpnp.utils import transforms


class PRDataset(Dataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        mask = self.masks[np.random.randint(0, len(self.masks))]
        sigma_n = 0

        if self.repeat > 1:
            index = index % len(self.fns)

        imgpath = join(self.datadir, self.fns[index])
        name = os.path.splitext(self.fns[index])[0]
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2, 0, 1))
        else:
            raise NotImplementedError

        target = torch.from_numpy(target).float()
        mask = torch.from_numpy(mask).float()

        y0 = transforms.complex_abs(
            transforms.cdp_forward(transforms.real2complex(target[None]), mask[None]))[0]

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        x0 = torch.ones_like(target)

        # convert to numpy array
        y0 = np.array(y0)
        target = np.array(target)
        x0 = np.array(x0)
        mask = np.array(mask)

        sigma_n = np.array(sigma_n)
        sigma_n = np.ones_like(x0) * sigma_n
        
        dic = {'y0': y0, 'x0': x0, 'output': x0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'name': name}

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class CPRDataset(Dataset):
    # compressive phase retrieval
    def __init__(self, datadir, fns, mask, samplematrix, noise_model=None, size=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]
        self.mask = mask
        self.samplematrix = samplematrix
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        mask = self.mask
        samplematrix = self.samplematrix
        sigma_n = 0

        if self.repeat > 1:
            index = index % len(self.fns)

        imgpath = join(self.datadir, self.fns[index])
        name = os.path.splitext(self.fns[index])[0]
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2, 0, 1))
        else:
            raise NotImplementedError

        target = torch.from_numpy(target).float()
        mask = torch.from_numpy(mask).float()

        y0 = transforms.complex_abs(
            transforms.cpr_forward(target[None], mask[None], samplematrix))[0]

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        x0 = torch.ones_like(target)

        # convert to numpy array
        y0 = np.array(y0)
        target = np.array(target)
        x0 = np.array(x0)
        mask = np.array(mask)

        sigma_n = np.array(sigma_n)
        sigma_n = np.ones_like(x0) * sigma_n
        
        dic = {'y0': y0, 'x0': x0, 'output': x0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'name': name}

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size
