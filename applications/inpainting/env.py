import torch
from tfpnp.env import PnPEnv
from tfpnp.data.batch import Batch


class HSIInpaintingEnv(PnPEnv):
    def __init__(self, data_loader, solver, max_step, device):
        super().__init__(data_loader, solver, max_step, device)
        
    def get_policy_state(self, state):
        return torch.cat([
            state.variables,
            state.low,
            state.Stx,
            state.mask,
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
            inputs = (variables, (state.Stx, state.mask))
            return inputs
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
    
    def _build_next_state(self, state, solver_state):
        return Batch(gt=state.gt, 
                     low=state.low, 
                     Stx=state.Stx, 
                     mask=state.mask,
                     variables=solver_state,
                     T=state.T)
        
    def _observation(self):
        idx_left = self.idx_left

        return Batch(gt=self.state['gt'][idx_left, ...], 
                     low=self.state['low'][idx_left, ...], 
                     Stx=self.state['Stx'][idx_left, ...], 
                     mask=self.state['mask'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])