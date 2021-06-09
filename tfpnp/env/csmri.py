import torch
from .base import PnPEnv
from ..pnp.util.transforms import complex2channel, complex2real

BOOL = True if float(torch.__version__[:3]) >= 1.3 else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSMRIEnv(PnPEnv):
    def __init__(self, data_loader, solver, max_step):
        super().__init__(data_loader, solver, max_step)
    
    def get_policy_state(self, state):
        x, z, u = state['solver']
        x = complex2real(x)
        z = complex2real(z)
        u = complex2real(u)
        
        y0 = complex2channel(state['y0'])
        ATy0 = complex2real(state['ATy0'])
        mask = state['mask']
        
        T = state['T']
        sigma_n = complex2real(state['sigma_n'])
        
        return torch.cat([x, z, u, y0, ATy0, mask, T, sigma_n], 1)
    
    def get_eval_state(self, state):
        return self.get_policy_state(state)