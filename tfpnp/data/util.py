import torch
import numpy as np

from PIL import Image
import math


def center_crop(img, target_size):
    # img: [H,W,C]
    w, h = target_size
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


def data_augment(img):
    C, H, W = img.shape    
    if np.random.randint(2, size=1)[0] == 1:  # random flip
        img = np.flip(img, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        img = np.flip(img, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        img = np.transpose(img, (0, 2, 1))        
    img = np.ascontiguousarray(img)
    return img
    

def dict_to_device(dic, device):
    dic = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in dic.items()}
    return dic