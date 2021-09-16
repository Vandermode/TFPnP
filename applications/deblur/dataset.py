import os

import numpy as np
import scipy
from scipy import ndimage
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset

from utils.dpir.utils_image import single2tensor4
from utils.dpir import utils_sisr as sr
from tfpnp.data.util import center_crop

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

class GaussianBlur():
    def __init__(self, ksize=8, sigma=3):
        self.k = fspecial_gaussian(ksize, sigma)
    
    def __call__(self, img):
        img_L = ndimage.filters.convolve(img, np.expand_dims(self.k, axis=2), mode='wrap')
        return img_L

    def kernel(self):
        return self.k
 
class GaussianNoise(object):
    def __init__(self, sigma):
        np.random.seed(seed=0)  # for reproducibility
        self.sigma = sigma

    def __call__(self, img):
        img_L = img + np.random.normal(0, self.sigma, img.shape)
        return img_L
    

class HSIDeblurDataset(Dataset):
    def __init__(self, datadir, training=True, target_size=None):
        self.datadir = datadir
        self.target_size = target_size
        self.fns = [im for im in os.listdir(self.datadir) if im.endswith(".mat")]  
        #TODO: the following line is important, if using entire dataset, training will be failed
        self.fns = self.fns[10:] if training else self.fns[:10] 
        self.blur = GaussianBlur()
        # self.awgn = GaussianNoise(10/255)
        
    def __getitem__(self, index):
        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = loadmat(imgpath)['gt']
        if self.target_size is not None:
            target = center_crop(target, self.target_size)
            
        gt = single2tensor4(target)
        low = self.blur(target)
        # low = self.awgn(low)
        k = np.expand_dims(self.blur.kernel(), 2)
        img_L_tensor, k_tensor = single2tensor4(low), single2tensor4(k)
        FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, 1)
        
        dic = {'low': img_L_tensor[0], 'FB': FB[0], 'FBC': FBC[0], 'F2B': F2B[0], 'FBFy': FBFy[0], 
               'gt': gt[0], 'output': img_L_tensor[0], 'name': self.fns[index][:-4]}

        # low, gt, output: [31,512,512]
        # k: [1,8,8]
        # FB,FBC,F2B: [1,512,512,2]
        # FBFy: [31,512,512,2]
        
        return dic
    
    def __len__(self):
        return len(self.fns)