import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from functools import partial


def psnr_b_max(gt, recon):
    h, w, c = gt.shape
    psnr = 0.
    for k in range(c):
        psnr += 10*np.log10(h*w*np.amax(gt[:,:,k])**2/sum(sum((recon[:,:,k]-gt[:,:,k])**2)))
    return psnr/c

def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam

def pnsr_qrnn3d(X, Y):
    return np.mean(cal_bwpsnr(X, Y))

def ssim_qrnn3d(X, Y):
    X = X.transpose(2,0,1)
    Y = Y.transpose(2,0,1)
    return np.mean(cal_bwssim(X, Y))

def sam_qrnn3d(X, Y):
    X = X.transpose(2,0,1)
    Y = Y.transpose(2,0,1)
    return cal_sam(X, Y)

class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = X[ch,:,:]
            y = Y[ch,:,:]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwssim = Bandwise(compare_ssim)
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))

def cal_sam(X, Y, eps=1e-8):
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)  
    tmp[np.argwhere(tmp>1)] = 1  
    return np.mean(np.real(np.arccos(tmp)))

def ergas(gt, pred, r=1):
    b = gt.shape[-1]
    ergas = 0
    for i in range(b):
        ergas += np.mean((gt[:,:,i]-pred[:,:,i])**2) / (np.mean(gt[:,:,i])**2)
    ergas = 100*r*np.sqrt(ergas/b)
    return ergas