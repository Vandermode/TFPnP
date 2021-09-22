from .solver.base import PnPSolver
from .denoiser import GRUNetDenoiser, UNetDenoiser2D, Denoiser


def create_denoiser(opt):
    print(f'[i] use denoiser: {opt.denoiser}')
    
    if opt.denoiser == 'unet':
        denoiser = UNetDenoiser2D()
    elif opt.denoiser == 'grunet':
        denoiser = GRUNetDenoiser()
    else:
        raise NotImplementedError

    return denoiser