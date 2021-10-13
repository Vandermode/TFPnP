import torch
import numpy as np

from tfpnp.pnp.solver.base import IADMMSolver, PGSolver
from tfpnp.utils import transforms


class CTMixin:
    def filter_aux_inputs(self, state):
        return (state['y0'], state['view'])


class IADMMSolver_CT(CTMixin, IADMMSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        self.radon_generator = transforms.RadonGenerator()        

    def forward(self, inputs, parameters, iter_num=None):
        # state: torch.cat([x, z, u], dim=1)
        # decode action
        variables, (y0, view) = inputs
        sigma_d, mu, tau = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        B = x.shape[0]
        
        radon = self.radon_generator(x.shape[-1], int(view[0,0,0,0].item() * 120))
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        for i in range(iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i]
            _tau = tau[:, i]

            # x step
            # xs.append(z - u)
            x = self.prox_mapping(z - u, _sigma_d)  # plug-and-play proximal mapping

            # z step (x + u)
            # inexact solution
            _tau = _tau.view(B, 1, 1, 1)
            _mu = _mu.view(B, 1, 1, 1)
            
            z = z - _tau * (radon.backprojection_norm(radon.forward(z) - y0) + _mu * (z - (x + u))) 

            # u step
            u = u + x - z

        next_variables = torch.cat([x, z, u], dim=1)

        return next_variables


class PGSolver_CT(CTMixin, PGSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        self.radon_generator = transforms.RadonGenerator()

    def forward(self, inputs, parameters, iter_num=None):
        variables, (y0, view) = inputs
        sigma_d, tau = parameters

        x = variables
        B = x.shape[0]
        
        radon = self.radon_generator(x.shape[-1], int(view[0,0,0,0].item() * 120))
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        for i in range(iter_num):
            _sigma_d = sigma_d[:, i]
            _tau = tau[:, i]

            # gradient descent
            _tau = _tau.view(B, 1, 1, 1)
            z = x - _tau * radon.backprojection_norm(radon.forward(x) - y0)
            
            # denoising
            x = self.prox_mapping(z, _sigma_d)
            
        next_variables = x

        return next_variables


_solver_map = {
    'iadmm': IADMMSolver_CT, 
    'pg': PGSolver_CT,    
}

def create_solver_ct(opt, denoiser):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser)
    else:
        raise NotImplementedError

    return solver
