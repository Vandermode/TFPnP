import torch
import numpy as np

from tfpnp.pnp.solver.base import IADMMSolver, PGSolver
from tfpnp.utils.transforms import complex2real, real2complex, complex_abs, cdp_backward, cdp_forward, fft2, ifft2


# decorator
def complex2real_decorator(func):
    def real_func(*args, **kwargs):
        return complex2real(func(*args, **kwargs))
    return real_func


class PRMixin:
    @complex2real_decorator
    def get_output(self, state):
        return super().get_output(state)

    def filter_aux_inputs(self, state):
        return (state['y0'], state['mask'])


class IADMMSolver_PR(PRMixin, IADMMSolver):
    #TODO warning: PRMixin must be put behind the ADMMSolver class
    def __init__(self, denoiser):
        super().__init__(denoiser)
        
    def reset(self, data):
        x = real2complex(data['x0'].clone().detach())
        z = x.clone().detach()
        u = torch.zeros_like(x)

        variables = torch.cat([x, z, u], dim=1)
        return variables

    def forward(self, inputs, parameters, iter_num=None):
        # state: torch.cat([x, z, u], dim=1)
        # decode action
        variables, (y0, mask) = inputs
        sigma_d, mu, tau = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        B = x.shape[0]
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        for i in range(iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i]
            _tau = tau[:, i]

            # x step
            # xs.append(z - u)            
            x = real2complex(self.prox_mapping(complex2real(z - u), _sigma_d))  # plug-and-play proximal mapping

            # z step (x + u)
            # inexact solution
            _tau = _tau.view(B, 1, 1, 1, 1)
            _mu = _mu.view(B, 1, 1, 1, 1)

            Az = cdp_forward(z, mask)   # Az
            y_hat = complex_abs(Az)     # |Az|
            meas_err = y_hat - y0
            gradient_forward = torch.stack((meas_err/y_hat*Az[...,0], meas_err/y_hat*Az[...,1]), -1)
            gradient = cdp_backward(gradient_forward, mask)
            z = z - _tau * (gradient + _mu * (z - (x + u)))

            # u step
            u = u + x - z

        next_variables = torch.cat([x, z, u], dim=1)

        return next_variables


class PGSolver_PR(PRMixin, PGSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        
    def reset(self, data):
        x = real2complex(data['x0'].clone().detach())
        variables = x
        return variables        

    def forward(self, inputs, parameters, iter_num=None):
        # state: x
        variables, (y0, mask) = inputs
        sigma_d, tau = parameters

        x = variables
        N = x.shape[0]

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        for i in range(iter_num):
            # gradient descent
            temp = fft2(x) - y0
            temp[~mask, :] = 0
            _tau = tau[:, i].view(N, 1, 1, 1, 1)
            z = x - _tau * ifft2(temp)

            # denoising
            x = real2complex(self.prox_mapping(complex2real(z), sigma_d[:, i]))

        next_variables = x

        return next_variables 


_solver_map = {
    'iadmm': IADMMSolver_PR, 
    'pg': PGSolver_PR,
}

def create_solver_pr(opt, denoiser):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser)
    else:
        raise NotImplementedError

    return solver
