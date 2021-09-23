import torch.nn as nn
import torch
from tfpnp.pnp.denoiser import Denoiser


class PnPSolver(nn.Module):
    def __init__(self, denoiser: Denoiser):
        super().__init__()
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

    def prox_mapping(self, x, sigma):
        return self.denoiser.denoise(x, sigma)

    @property
    def num_var(self):
        raise NotImplementedError

    def filter_aux_inputs(self, state):
        raise NotImplementedError

    def filter_hyperparameter(self, action):
        raise NotImplementedError


class ADMMSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
    
    @property
    def num_var(self):
        return 3
        
    def reset(self, data):
        x = data['x0'].clone().detach() # [B,1,W,H,2]
        z = x.clone().detach()          # [B,1,W,H,2]
        u = torch.zeros_like(x)         # [B,1,W,H,2]
        return torch.cat((x, z, u), dim=1)        
    
    def get_output(self, state):
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu']


class IADMMSolver(ADMMSolver):
    # Inexact ADMM
    def __init__(self, denoiser):
        super().__init__(denoiser)
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu'], action['tau']

class HQSSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 2

    def reset(self, data):
        x = data['x0'].clone().detach()
        z = x.clone().detach()
        variables = torch.cat([x, z], dim=1)

        return variables

    def get_output(self, state):
        x, _, = torch.split(state, state.shape[1] // 2, dim=1)
        return x
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu']

class PGSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 1

    def reset(self, data):
        x = data['x0'].clone().detach()
        variables = x
        return variables

    def get_output(self, state):
        x = state
        return x
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['tau']

class APGSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        import numpy as np
        self.qs = np.zeros(30)
        q = 1
        for i in range(30):
            self.qs[i] = q
            q_prev = q
            q = (1 + (1 + 4 * q_prev**2)**(0.5)) / 2

    @property
    def num_var(self):
        return 2

    def reset(self, data):
        x = data['x0'].clone().detach()
        s = x.clone().detach()
        variables = torch.cat([x, s], dim=1)
        return variables

    def get_output(self, state):
        x, _, = torch.split(state, state.shape[1] // 2, dim=1)
        return x    

    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['tau'], action['beta']


class REDADMMSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 3

    def reset(self, data):
        x = data['x0'].clone().detach()
        z = x.clone().detach()
        u = torch.zeros_like(x)
        variables = torch.cat([x, z, u], dim=1)
        return variables

    def get_output(self, state):
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x

    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu'], action['lamda']


class AMPSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 2        

    def reset(self, data):
        z = data['y0'].clone().detach()
        x = torch.zeros_like(data['x0'])
        variables = torch.cat([x, z], dim=1)

        return variables

    def get_output(self, state):
        x, _ = torch.split(state, state.shape[1] // 2, dim=1)
        return x

    def filter_hyperparameter(self, action):
        return action['sigma_d']
