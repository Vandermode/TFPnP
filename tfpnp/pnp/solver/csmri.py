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
        return torch.cat((x, z, u), dim=1)

    def forward(self, inputs, parameters, iter_num=None):    
        # y0:    [B,1,W,H,2]
        # mask:  [B,1,W,H] 
        # x,z,u: [B,1,W,H,2]
        
        variables, (y0, mask) = inputs
        mu, sigma_d = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = mu.shape[-1]
        
        for i in range(iter_num):
            # x step
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z - u), sigma_d[:, i]))  # plug-and-play proximal mapping

            # z step
            z = transforms.fft2(x + u)

            _mu = mu[:, i].view(x.shape[0], 1, 1, 1, 1)
            temp = ((_mu * z.clone()) + y0) / (1 + _mu)
            z[mask, :] =  temp[mask, :]

            z = transforms.ifft2(z)

            # u step
            u = u + x - z
        
        return torch.cat((x, z, u), dim=1)
    
    def get_output(self, state):
        # just return x after convert to real
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return transforms.complex2real(x)
    
    def filter_additional_input(self, state):
        return (state['y0'], state['mask'])
    
    def filter_hyperparameter(self, action):
        return action['mu'], action['sigma_d']