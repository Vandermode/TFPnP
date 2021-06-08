import torch

from .base import PnPSolver
from ..denoiser import Denoiser
from ..util import transforms

class ADMMSolver_CSMRI(PnPSolver):
    def __init__(self, denoiser: Denoiser):
        super().__init__(denoiser)
        
    def reset(self, data):
        x = data['x0'].clone().detach() # [B,1,W,H,2]
        z = x.clone().detach()          # [B,1,W,H,2]
        u = torch.zeros_like(x)         # [B,1,W,H,2]
        
        self.mask = data['mask'].unsqueeze(1).bool() # [B,1,W,H]
        self.y0 = data['y0'] # [B,1,W,H,2]
        
        return (x,z,u)

    def forward(self, inputs, parameters, iter_num):    
        x, z, u = inputs
        mu, sigma_d = parameters

        for i in range(iter_num):
            # x step
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z - u), sigma_d[:, i]))  # plug-and-play proximal mapping

            # z step
            z = transforms.fft2(x + u)

            _mu = mu[:, i].view(x.shape[0], 1, 1, 1, 1)
            temp = ((_mu * z.clone()) + self.y0) / (1 + _mu)
            z[self.mask, :] =  temp[self.mask, :]

            z = transforms.ifft2(z)

            # u step
            u = u + x - z
        
        return x, z, u