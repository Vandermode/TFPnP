import torch.nn as nn
import torch

class PnPSolver(nn.Module):
   
    def reset(self, data):
        """ Reset the internal states according to the input data

        Args:
            data: a object contains the input data

        Returns:
            states: a object contains the internal states of the solver
        """
        raise NotImplementedError
    
    def forward(self, inputs, parameters, iter_num):
        """ Iteratively solve the problem via pnp algorithm for `self.iter_num` steps.

        Args:
            inputs: an object contains the previous internal states
            parameters: an object contains hyperparameters for the subsequent iteration
            iter_num: number of iteration
            
        Returns:
            states: a object contains the current internal states of the solver
        """
        raise NotImplementedError
    
    def get_output(self, state):
        """ Get output from intermediate state

        Args:
            state: a object contains the current internal states of the solver
            
        Returns:
            output: restored image
        """
        raise NotImplementedError
    
    @property
    def num_var(self):
        raise NotImplementedError
    
    def filter_additional_input(self, state):
        raise NotImplementedError
    
    def filter_hyperparameter(self, action):
        raise NotImplementedError
    
    
class ADMMSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
    
    @property
    def num_var(self):
        return 3
    
    def reset(self, data):
        x = data['input'].clone().detach()
        v = x.clone().detach()
        u = torch.zeros_like(x)
        return torch.cat((x, v, u), dim=1)
    
    def get_output(self, state):
        # just return x after convert to real
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x
    
    def filter_hyperparameter(self, action):
        return action['mu'], action['sigma_d']