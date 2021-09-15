import numpy as np
from scipy.io import loadmat
import os

from solver import Interpolation_OLRT_3D

from tfpnp.data.dataset import ImageDir
from tfpnp.data.util import center_crop
from utils.dpir.utils_image import single2tensor4

class FastHyStripe:
    """ Input: [W,H,B] """
    def __init__(self, num_bands=15, bandwise=False):
        self.num_bands = num_bands # how many bands will be added stripe noise
        self.bandwise = bandwise # if the location of stripe noise are the same

    def __call__(self, img):
        import time
        np.random.seed(int(time.time()))
        mask = np.ones_like(img)
        w, h, b = img.shape
        # random select 4 band to add stripe (actually dead line)
    
        start_band = 10
        # start_band = 0
        for i in range(start_band,start_band+self.num_bands):
            stripes = np.random.choice(h, 20, replace=False)
            for k, j in enumerate(stripes):
                t = np.random.rand()
                if k == 4:
                    mask[:, j:j+30, i] = 0
                elif k == 10:
                    mask[:, j:j+15, i] = 0
                elif t > 0.6:
                    mask[:, j:j+4, i] = 0
                else:
                    mask[:, j:j+2, i] = 0
            if self.bandwise:
                break
        if self.bandwise:
            mask[:,:,start_band:start_band+self.num_bands] = np.expand_dims(mask[:,:,start_band],axis=-1)
        img_L = img * mask
        return img_L, mask

class GaussianNoise(object):
    def __init__(self, sigma):
        np.random.seed(seed=0)  # for reproducibility
        self.sigma = sigma

    def __call__(self, img):
        img_L = img + np.random.normal(0, self.sigma, img.shape)
        return img_L
    

class HSIInpaintingDataset(ImageDir):
    def __init__(self, datadir, training, target_size):
        super().__init__(datadir, training=training)
        self.target_size = target_size
        self.degrade = FastHyStripe()
        self.awgn = GaussianNoise(30/255)
        
    def _get_data(self, path):
        target = loadmat(path)['gt']
        if self.target_size is not None:
            target = center_crop(target, self.target_size)
            
        gt = single2tensor4(target)
        
        low = target
        low, mask = self.degrade(low)
        low = self.awgn(low)
        
        # x = Interpolation_OLRT_3D(low, mask)
        low = single2tensor4(low)
        x = low
        
        mask = single2tensor4(mask.astype('float'))
        Stx = x * mask
        
        name = os.path.basename(path)[:-4]
        
        dic = {'low': low[0], 'Stx': Stx[0], 'mask': mask[0], 'input': x[0],
               'gt': gt[0], 'output': x[0], 'name': name}
        # all [31, 128, 128]
        return dic