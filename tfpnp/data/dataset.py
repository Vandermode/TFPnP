from torch.functional import Tensor
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
import numpy as np
import os
import torch 
from PIL import Image
import math

from .degrade import GaussianBlur
from ..pnp.util.dpir.utils_image import single2tensor4
from ..pnp.util.dpir import utils_sisr as sr
from ..pnp.util import transforms
from ..pnp.util.transforms import RadonGenerator, complex2real

def center_crop(img, target_size):
    # img: [H,W,C]
    w,h = target_size
    wo, ho, _ = img.shape
    return img[(wo-w)//2:(wo-w)//2+w, (ho-h)//2:(ho-h)//2+h, :]

def scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)
 

def dict_to_device(dic, device):
    dic = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in dic.items()} 
    return dic
 
 ##############################################
 #   HSI Deblur dataset                          
 ##############################################
 
 
class HSIDeblurDataset(Dataset):
    def __init__(self, datadir, training=True, target_size=None):
        self.datadir = datadir
        self.target_size = target_size
        self.fns = [im for im in os.listdir(self.datadir) if im.endswith(".mat")]  
        self.fns = self.fns[-10:] if training else self.fns[:-10]
        self.blur = GaussianBlur()
        
    def __getitem__(self, index):
        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = loadmat(imgpath)['gt']
        if self.target_size is not None:
            target = center_crop(target, self.target_size)
            
        gt = single2tensor4(target)
        low = self.blur(target)
        k = np.expand_dims(self.blur.kernel(), 2)
        img_L_tensor, k_tensor = single2tensor4(low), single2tensor4(k)
        FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, 1)
        
        dic = {'low': img_L_tensor[0], 'FB': FB[0], 'FBC': FBC[0], 'F2B': F2B[0], 'FBFy': FBFy[0], 'gt': gt[0]}
        
        # low, gt, output: [1,31,512,512]
        # k: [1,1,8,8]
        # FB,FBC,F2B: [1,1,512,512,2]
        # FBFy: [1,31,512,512,2]
        
        return dic
    
    def __len__(self):
        return len(self.fns)


 ##############################################
 #  CSMRI dataset                          
 ##############################################
 

BOOL = True if float(torch.__version__[:3]) >= 1.3 else False

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
        mask = self.masks[np.random.randint(0, len(self.masks))]

        if BOOL:
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
        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'output': output}

        # y0,x0,ATy0, sigma_n: C, W, H, 2
        # gt, output: C, W, H
        # mask: W, H
        
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

        return mat
        
    def __len__(self):
        return len(self.fns)
