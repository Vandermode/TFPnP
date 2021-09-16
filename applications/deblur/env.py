import torch
from tfpnp.env import PnPEnv
from tfpnp.utils.transforms import complex2channel, complex2real, real2complex
from tfpnp.data.batch import Batch


class DeblurEnv(PnPEnv):
    def __init__(self, data_loader, solver, max_step, device):
        super().__init__(data_loader, solver, max_step, device)
        
    def get_policy_state(self, state):
        return torch.cat([
            state.variables,
            state.low,
            complex2channel(state.FB),
            complex2channel(state.FBFy),
            state.T,
        ], 1)
    
    def get_eval_state(self, state):
        return self.get_policy_state(state)
    
    def _get_attribute(self, state, key):        
        if key == 'gt':
            return state.gt
        elif key == 'output':
            return self.solver.get_output(state.variables)
        elif key == 'input':
            return state.low
        elif key == 'solver_input':
            variables = state.variables
            inputs = (variables, (state.FB, state.FBC, state.F2B, state.FBFy))
            return inputs
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
    
    def _build_next_state(self, state, solver_state):
        return Batch(gt=state.gt, 
                     low=state.low, 
                     FB=state.FB, 
                     FBC=state.FBC,
                     F2B=state.F2B,
                     FBFy=state.FBFy,
                     variables=solver_state,
                     T=state.T)
        
    def _observation(self):
        idx_left = self.idx_left

        return Batch(gt=self.state['gt'][idx_left, ...], 
                     low=self.state['low'][idx_left, ...], 
                     FB=self.state['FB'][idx_left, ...], 
                     FBC=self.state['FBC'][idx_left, ...],
                     F2B=self.state['F2B'][idx_left, ...],
                     FBFy=self.state['FBFy'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])