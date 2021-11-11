import os
import numpy as np
from PIL import Image

import torch 
from torch.utils.data.dataset import Dataset

from tfpnp.data.util import scale_height, scale_width
from tfpnp.utils import transforms


# Only for testing, do not use it for training!
class CTDataset(Dataset):
    def __init__(self, datadir, fns, view, noise_model=None, size=None, target_size=None, repeat=1):
        super(CTDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.view = view
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.radon_generator = transforms.RadonGenerator()

    def __getitem__(self, index):
        view = self.view
        sigma_n = 0

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
        
        # we use torch_radon to implement CT forward model, which should be executed on GPU
        target = torch.from_numpy(target).cuda()
        
        resolution = target.shape[-1]
        radon = self.radon_generator(resolution, view)

        # y0 : sinogram
        y0 = radon.forward(target)

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        ATy0 = radon.backprojection_norm(y0)
        x0 = radon.filter_backprojection(y0)
        output = ATy0.clone().detach()
                
        view = torch.ones_like(target) * view / 120
        sigma_n = torch.ones_like(target) * sigma_n
        
        dic = {'y0': y0, 'ATy0': ATy0, 'output': output, 'x0': x0, 'gt': target, 'view': view, 'sigma_n': sigma_n, 'name': name}

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class CT_transform:
    def __init__(self, view, noise_model):
        self.view = view
        self.noise_model = noise_model
        self.radon_generator = transforms.RadonGenerator()        
    
    def __call__(self, x):        
        x = x.cuda()
        resolution = x.shape[-1]
        radon = self.radon_generator(resolution, self.view)

        # y0 : sinogram
        y0 = radon.forward(x)

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        ATy0 = radon.backprojection_norm(y0)
        x0 = radon.filter_backprojection(y0)
        output = ATy0.clone().detach()
                
        view = torch.ones_like(x) * self.view / 120
        sigma_n = torch.ones_like(x) * sigma_n
        
        dic = {'y0': y0, 'ATy0': ATy0, 'output': output, 'x0': x0, 'gt': x, 'view': view, 'sigma_n': sigma_n}
        return dic
