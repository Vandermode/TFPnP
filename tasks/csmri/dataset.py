import os
import numpy as np
from PIL import Image

import torch 
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat

from tfpnp.data.util import scale_height, scale_width
from tfpnp.utils import transforms
from tfpnp.utils.transforms import complex2real


class CSMRIDataset(Dataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1):
        super(CSMRIDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        mask = self.masks[1]
        mask = mask.astype(np.bool)
        
        sigma_n = 0

        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size            
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)        

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2,0,1))
        else:
            raise NotImplementedError

        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)
        
        y0 = transforms.fft2(torch.stack([target, torch.zeros_like(target)], dim=-1))
        # y0[:, ~mask, :] = 0

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)
            
        y0[:, ~mask, :] = 0

        ATy0 = transforms.ifft2(y0)
        x0 = ATy0.clone().detach()

        output = complex2real(ATy0.clone().detach())
        mask = mask.unsqueeze(0).bool()
        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'output': output, 'input': x0}

        # y0,x0,ATy0, sigma_n: C, W, H, 2
        # gt, output: C, W, H
        # mask: 1, W, H
        
        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class CSMRIEvalDataset(Dataset):
    def __init__(self, datadir, fns=None):
        super(CSMRIEvalDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".mat")]             
    
    def __getitem__(self, index):
        fn = self.fns[index]
        mat = loadmat(os.path.join(self.datadir, fn))

        mat['name'] = mat['name'].item()
        mat.pop('__globals__', None)
        mat.pop('__header__', None)
        mat.pop('__version__', None)
        mat['output'] = complex2real(mat['ATy0'])
        mat['input'] = mat['x0']
        mat['mask'] = np.expand_dims(mat['mask'], axis=0).astype('bool')
        
        return mat
        
    def __len__(self):
        return len(self.fns)
