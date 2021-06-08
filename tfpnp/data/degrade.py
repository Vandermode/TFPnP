import numpy as np
import scipy
from scipy import ndimage

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