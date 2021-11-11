import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from .util import scale_height, scale_width


class ImageFolder(data.dataset.Dataset):
    def __init__(self, datadir, fns=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]        
        self.target_size = target_size
        self.repeat = repeat

    def __getitem__(self, index):
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
        
        return target

    def __len__(self):
        return len(self.fns) * self.repeat
