import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch 
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from tfpnp.data.util import scale_height, scale_width
from tfpnp.utils import transforms


class SPIDataset(Dataset):
    def __init__(self, datadir, fns, Ks, size=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.Ks = Ks  # oversampling rates
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        K = self.Ks[np.random.randint(0, len(self.Ks))]

        if self.repeat > 1:
            index = index % len(self.fns)
        
        imgpath = os.path.join(self.datadir, self.fns[index])
        name = os.path.splitext(self.fns[index])[0]
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
        
        with torch.no_grad():
            target = torch.from_numpy(target).float()
            y0 = transforms.spi_forward(target[None], K, K**2, 1)
            x0 = F.avg_pool2d(y0, K)  # simple average

        y0 = y0[0]
        x0 = x0[0]        
        output = x0.clone().detach()
        # K = torch.ones_like(target) * K / 10.  # [0-1]
        
        # convert to numpy array
        y0 = np.array(y0)
        target = np.array(target)
        x0 = np.array(x0)
        K = np.array(K)        
        K = np.ones_like(target) * K / 10

        # dic = {'y0': y0, 'x0': x0, 'gt': target, 'K': K, 'name': name}
        # dic = {'x0': x0, 'output': x0, 'gt': target, 'K': K, 'name': name}
        dic = {'x0': x0, 'output': output, 'gt': target, 'K': K}

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class SPIEvalDataset(Dataset):
    def __init__(self, datadir, fns=None):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".mat")]      
    
    def __getitem__(self, index):
        fn = self.fns[index]
        mat = loadmat(os.path.join(self.datadir, fn))

        mat['name'] = mat['name'].item()
        mat.pop('__globals__', None)
        mat.pop('__header__', None)
        mat.pop('__version__', None)        
        mat['output'] = mat['x0']
        mat['input'] = mat['x0']
        mat['K'] = (np.ones_like(mat['gt']) * mat['K'].reshape(1, 1, 1) / 10.).astype(np.float32)  # [0-1]

        return mat
        
    def __len__(self):
        return len(self.fns)
