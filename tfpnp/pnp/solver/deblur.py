import torch

from .base import PnPSolver
from ..denoiser import Denoiser
from ..util.dpir import utils_sisr as sr
from ..util import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ADMMSolver_Deblur(PnPSolver):
    def __init__(self, denoiser: Denoiser):
        super().__init__(denoiser)
    
    @property
    def num_var(self):
        return 3
    
    def reset(self, data):
        x = data['low'].clone().detach()
        v = x.clone().detach()
        u = torch.zeros_like(x)
        return torch.cat((x, v, u), dim=1)
    
    def forward(self, inputs, parameters, iter_num=None):
    
        variables, (FB, FBC, F2B, FBFy) = inputs
        rhos, sigmas = parameters
        
        x, v, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = rhos.shape[-1]
            
        for i in range(iter_num):
            # --------------------------------
            # step 1, FFT
            # --------------------------------

            xtilde = v - u
           
            tau = rhos[:,i].float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = sr.data_solution_admm_sr(xtilde.float(), FB, FBC, F2B, FBFy, tau, 1)

            # --------------------------------
            # step 2, denoiser
            # --------------------------------

            vtilde = x + u

            v = self.denoiser.denoise(vtilde, sigmas[:,i])
            
            # --------------------------------
            # step 3, u subproblem
            # --------------------------------

            u = u + x - v
            
        return torch.cat((x, v, u), dim=1)
    
    def get_output(self, state):
        # just return x after convert to real
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x
    
    def filter_additional_input(self, state):
        return (state['FB'], state['FBC'], state['F2B'], state['FBFy'])
    
    def filter_hyperparameter(self, action):
        return action['mu'], action['sigma_d']