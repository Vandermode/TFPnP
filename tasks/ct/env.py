
import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv


class CTEnv(PnPEnv):
    # class attribute: the dimension of ob (exclude solver variable)
    ob_base_dim = 4
    def __init__(self, data_loader, solver, max_episode_step, device, data_transform=None):        
        super().__init__(data_loader, solver, max_episode_step, device, data_transform)
    
    def get_policy_ob(self, ob):
        return torch.cat([
            ob.variables,
            ob.ATy0,
            ob.view,
            ob.T,
            ob.sigma_n
        ], 1)
    
    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)
    
    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.ATy0
        elif key == 'solver_input':
            return (ob.variables, (ob.y0, ob.view))
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     y0=ob.y0,
                     ATy0=ob.ATy0,
                     variables=solver_state,
                     view=ob.view,
                     sigma_n=ob.sigma_n,                     
                     T=ob.T + 1/self.max_episode_step)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     y0=self.state['y0'][idx_left, ...],
                     ATy0=self.state['ATy0'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     view=self.state['view'][idx_left, ...],
                     sigma_n=self.state['sigma_n'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])
