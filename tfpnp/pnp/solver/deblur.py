import torch

from .base import PnPSolver
from ..denoiser import Denoiser
from ..util.dpir import utils_sisr as sr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ADMMSolver_Deblur(PnPSolver):
    def __init__(self, denoiser: Denoiser):
        super().__init__(denoiser)
        
    def reset(self, data):
        x = data['low'].clone().detach()
        v = x.clone().detach()
        u = torch.zeros_like(x)

        self.FB = data['FB']
        self.FBC = data['FBC']
        self.F2B = data['F2B']
        self.FBFy = data['FBFy']
        
        return (x, v, u)
    
    def forward(self, inputs, parameters, iter_num):
        """
            img_L: [W, H, C], range = [0,1]
        """
        x, v, u = inputs
        rhos, sigmas = parameters
        
        for i in range(iter_num):
            # --------------------------------
            # step 1, FFT
            # --------------------------------

            xtilde = v - u
           
            tau = rhos[:,i].float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = sr.data_solution_admm_sr(xtilde.float(), self.FB, self.FBC, self.F2B, self.FBFy, tau, 1)

            # --------------------------------
            # step 2, denoiser
            # --------------------------------

            vtilde = x + u

            v = self.denoiser.denoise(vtilde, sigmas[:,i])
            
            # --------------------------------
            # step 3, u subproblem
            # --------------------------------

            u = u + x - v
            
        return (x, v, u)