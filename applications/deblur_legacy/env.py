import torch
from tfpnp.env import PnPEnv
from tfpnp.utils.transforms import complex2channel, complex2real, real2complex

class DeblurEnv(PnPEnv):
    def __init__(self, data_loader, solver, max_step, device):
        super().__init__(data_loader, solver, max_step, device)
        
    def get_policy_state(self, state):
        c = 31
        
        variables = state[:,c:4*c,...]
        low = state[:,4*c:5*c,...]
        FB = state[:,5*c:5*c+1,...]
        FBFy = state[:,5*c+3:5*c+3+c,...]
        T = state[:,5*c+3+c:5*c+3+c+1,...]
        
        return torch.cat([
            complex2real(variables),
            complex2real(low),
            complex2channel(FB),
            complex2channel(FBFy),
            complex2real(T),
        ], 1)
    
    def get_eval_state(self, state):
        return self.get_policy_state(state)
    
    def _get_attribute(self, state, key):
        c = 31
        
        if key == 'gt':
            return complex2real(state[:, :c, ...])
        elif key == 'output':
            return complex2real(state[:, c:2*c, ...])
        elif key == 'input':
            return complex2real(state[:,4*c:5*c,...])
        elif key == 'solver_input':
            variables = complex2real(state[:, c:4*c, ...])
            FB = state[:,5*c:5*c+1,...]
            FBC = state[:,5*c+1:5*c+2,...]
            F2B = state[:,5*c+2:5*c+3,...]
            FBFy = state[:,5*c+3:5*c+3+c,...]
            inputs = (variables, (FB, FBC, F2B, FBFy))
            return inputs
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
    
    def _build_next_state(self, state, solver_state):
        c = 31
        
        gt = state[:, :c, ...]
        variables = real2complex(solver_state)
        low = state[:,4*c:5*c,...]
        FB = state[:,5*c:5*c+1,...]
        FBC = state[:,5*c+1:5*c+2,...]
        F2B = state[:,5*c+2:5*c+3,...]
        FBFy = state[:,5*c+3:5*c+3+c,...]
        T = state[:,5*c+3+c:5*c+3+c+1,...]
        
        ob = torch.cat([gt, variables, low, FB, FBC, F2B, FBFy, T], 1)

        return ob
        
    def _observation(self):
        idx_left = self.idx_left
        
        gt = real2complex(self.state['gt'][idx_left, ...])
        low = real2complex(self.state['low'][idx_left, ...])
        FB = self.state['FB'][idx_left, ...] # B, 1, W, H, 2
        FBC = self.state['FBC'][idx_left, ...] # B, 1, W, H, 2
        F2B = self.state['F2B'][idx_left, ...] # B, 1, W, H, 2
        FBFy = self.state['FBFy'][idx_left, ...] # B, 31, W, H, 2
        variables = real2complex(self.state['solver'][idx_left, ...]) 
        T = real2complex(self.state['T'][idx_left, ...])  # B, 1, W, H, 2
     
        ob = torch.cat([gt, variables, low, FB, FBC, F2B, FBFy, T], 1)

        return ob