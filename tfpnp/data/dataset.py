from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
import numpy as np
import os

from .degrade import GaussianBlur
from ..util.dpir.utils_image import single2tensor4
from ..util.dpir import utils_sisr as sr

def center_crop(img, target_size):
    # img: [H,W,C]
    w,h = target_size
    wo, ho, _ = img.shape
    return img[(wo-w)//2:(wo-w)//2+w, (ho-h)//2:(ho-h)//2+h, :]
   
 
class HSIDeblurDataset(Dataset):
    def __init__(self, datadir, training=True, target_size=None, device=None):
        self.device = device
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
        
        if self.device is not None:
            dic = {k: v.to(self.device) for k, v in dic.items()}
        
        # low, gt, output: [1,31,512,512]
        # k: [1,1,8,8]
        # FB,FBC,F2B: [1,1,512,512,2]
        # FBFy: [1,31,512,512,2]
        
        return dic
    
    def __len__(self):
        return len(self.fns)