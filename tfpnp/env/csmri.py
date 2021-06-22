import torch
from .base import PnPEnv
from ..pnp.util.transforms import complex2channel, complex2real, real2complex

BOOL = True if float(torch.__version__[:3]) >= 1.3 else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSMRIEnv(PnPEnv):
    def __init__(self, data_loader, solver, max_step):
        super().__init__(data_loader, solver, max_step)
    
    def get_policy_state(self, state):
        num_var = self.solver.num_var
        
        variables = state[:, 1:num_var+1, ...]
        y0 = state[:, num_var+1:num_var+2, ...]
        ATy0 = state[:, num_var+2:num_var+3, ...]
        mask = state[:, num_var+3:num_var+4, ...]
        T = state[:, num_var+4:num_var+5, ...]
        sigma_n = state[:, num_var+5:num_var+6, ...]
        
        return torch.cat([
            complex2real(variables),
            complex2channel(y0),
            complex2real(ATy0),
            complex2real(mask),
            complex2real(T),
            complex2real(sigma_n),
        ], 1)
    
    def get_eval_state(self, state):
        return self.get_policy_state(state)
    
    def _observation(self):
        idx_left = self.idx_left

        gt = real2complex(self.state['gt'][idx_left, ...])
        y0 = self.state['y0'][idx_left, ...]
        ATy0 = self.state['ATy0'][idx_left, ...]
        variables = self.state['solver'][idx_left, ...]
        mask = real2complex(self.state['mask'][idx_left, ...]).float()
        sigma_n = self.state['sigma_n'][idx_left, ...]
        T = self.state['T'][idx_left, ...]

        ob = torch.cat([gt, variables, y0, ATy0, mask, T, sigma_n], 1)
        return ob
