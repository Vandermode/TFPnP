
import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv


class SPIEnv(PnPEnv):
    # class attribute: the dimension of ob (exclude solver variable)
    ob_base_dim = 3  
    def __init__(self, data_loader, solver, max_episode_step, device):
        super().__init__(data_loader, solver, max_episode_step, device)
    
    def get_policy_ob(self, ob):
        return torch.cat([
            ob.variables,
            ob.x0,
            ob.K,
            ob.T,
        ], 1)
    
    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)
    
    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.x0
        elif key == 'solver_input':
            return (ob.variables, (ob.x0, ob.K))
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     x0=ob.x0,
                     variables=solver_state,
                     K=ob.K,
                     T=ob.T + 1/self.max_episode_step)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     x0=self.state['x0'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     K=self.state['K'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])
