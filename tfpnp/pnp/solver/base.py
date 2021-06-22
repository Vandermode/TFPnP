import torch
import torch.nn as nn
from ..denoiser import Denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PnPSolver(nn.Module):
    def __init__(self, denoiser: Denoiser):
        super(PnPSolver, self).__init__()        
        self.denoiser = denoiser    

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