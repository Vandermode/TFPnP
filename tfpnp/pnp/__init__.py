from .solver.base import PnPSolver
from .denoiser import UNetDenoiser2D


def create_denoiser(opt):
    print(f'[i] use denoiser: {opt.denoiser}')
    
    if opt.denoiser == 'unet':
        denoiser = UNetDenoiser2D()
    else:
        raise NotImplementedError

    return denoiser