import ipdb
import torch
import numpy as np

from tfpnp.pnp.solver.base import ADMMSolver, HQSSolver, PGSolver, APGSolver, REDADMMSolver, AMPSolver
from tfpnp.utils import transforms


# decorator
def complex2real(func):
    def real_func(*args, **kwargs):
        return transforms.complex2real(func(*args, **kwargs))
    return real_func


class CSMRIMixin:
    @complex2real
    def get_output(self, state):
        return super().get_output(state)

    def filter_aux_inputs(self, state):
        return (state['y0'], state['mask'])


class ADMMSolver_CSMRI(CSMRIMixin, ADMMSolver):
    #TODO warning: CSMRIMixin must be put behind the ADMMSolver class
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):    
        # y0:    [B,1,W,H,2]
        # mask:  [B,1,W,H]
        # x,z,u: [B,1,W,H,2]
        
        variables, (y0, mask) = inputs
        sigma_d, mu = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = mu.shape[-1]
        
        for i in range(iter_num):
            # x step
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z - u), sigma_d[:, i]))  # plug-and-play proximal mapping

            # z step
            z = transforms.fft2(x + u)
            _mu = mu[:, i].view(x.shape[0], 1, 1, 1, 1)
            temp = ((_mu * z.clone()) + y0) / (1 + _mu)
            z[mask, :] =  temp[mask, :]
            z = transforms.ifft2(z)

            # u step
            u = u + x - z
        
        return torch.cat((x, z, u), dim=1)


class HQSSolver_CSMRI(CSMRIMixin, HQSSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):
        # state: torch.cat([x, z], dim=1)
        variables, (y0, mask) = inputs
        sigma_d, mu = parameters

        x, z = torch.split(variables, variables.shape[1] // 2, dim=1)
        N = x.shape[0]

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        for i in range(iter_num):
            # x step
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z), sigma_d[:, i]))  # plug-and-play proximal mapping

            # z step
            z = transforms.fft2(x)
            _mu = mu[:, i].view(N, 1, 1, 1, 1)
            temp = ((_mu * z.clone()) + y0) / (1 + _mu)
            z[mask, :] =  temp[mask, :]
            z = transforms.ifft2(z)

        next_variables = torch.cat([x, z], dim=1)
        
        return next_variables


class PGSolver_CSMRI(CSMRIMixin, PGSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

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
            temp = transforms.fft2(x) - y0
            temp[~mask, :] = 0
            _tau = tau[:, i].view(N, 1, 1, 1, 1)
            z = x - _tau * transforms.ifft2(temp)

            # denoising
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z), sigma_d[:, i]))

        next_variables = x

        return next_variables 


class APGSolver_CSMRI(CSMRIMixin, APGSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):
        # state: torch.cat([x, s], dim=1)
        variables, (y0, mask) = inputs
        sigma_d, tau, beta = parameters

        x, s = torch.split(variables, variables.shape[1] // 2, dim=1)
        N = x.shape[0]

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        # ind = (stepnum * self.num_loops).long()
        # q = self.qs[ind]

        for i in range(iter_num):
            _tau = tau[:, i].view(N, 1, 1, 1, 1)
            _beta = beta[:, i].view(N, 1, 1, 1, 1)

            # gradient descent
            temp = transforms.fft2(s) - y0
            temp[~mask, :] = 0
            z = s - _tau * transforms.ifft2(temp)
            
            # denoising
            x_prev = x
            x = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(z), sigma_d[:, i]))  

            # q update
            # q_prev = q
            # q = (1 + (1 + 4 * q_prev**2)**(0.5)) / 2
            
            # s update (extrapolation)            
            # s = x + ((q_prev - 1) / q).view(N, 1, 1, 1, 1) * (x - x_prev)
            s = x + _beta * (x - x_prev)

        next_variables = torch.cat([x, s], dim=1)

        return next_variables


class REDADMMSolver_CSMRI(CSMRIMixin, REDADMMSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):
        # state: torch.cat([x, z, u], dim=1)
        variables, (y0, mask) = inputs
        sigma_d, mu, lamda = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        N = x.shape[0]

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]
            
        for i in range(iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i].view(N, 1, 1, 1, 1)
            _lamda = lamda[:, i].view(N, 1, 1, 1, 1)

            # x step
            x_half = transforms.real2complex(self.denoiser.denoise(transforms.complex2real(x), _sigma_d))
            x = (_lamda * x_half + _mu * (z - u)) / (_mu + _lamda)

            # z step
            z = transforms.fft2(x + u)
            temp = ((_mu * z.clone()) + y0) / (1 + _mu)
            z[mask, :] =  temp[mask, :]
            z = transforms.ifft2(z)

            # u step
            u = u + x - z

        next_variables = torch.cat([x, z, u], dim=1)

        return next_variables     


class AMPSolver_CSMRI(CSMRIMixin, AMPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):
        # state: x
        variables, (y0, mask) = inputs
        sigma_d = parameters

        x, z = torch.split(variables, variables.shape[1] // 2, dim=1)

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        B = x.shape[0]
        M = mask.view(B, -1).sum(dim=-1).float()
        N = torch.tensor(mask.shape[-1] * mask.shape[-2]).float()

        for i in range(iter_num):            
            r = x + transforms.ifft2(z)
            r = transforms.complex2real(r)

            _sigma_d = transforms.complex_norm(z) / torch.sqrt(N)
            _sigma_d = _sigma_d * sigma_d[:, i]
            # _sigma_d = sigma_d[:, i]

            x = transforms.real2complex(self.denoiser.denoise(r, _sigma_d))

            epsilon = r.max() / 1000 + 1e-8
            delta = torch.randn_like(r)
            div_r = (self.prox_fun(r + delta*epsilon, _sigma_d) - transforms.complex2real(x))
            div_r = (delta * div_r).view(B, -1).sum(dim=-1) / epsilon

            o = z * div_r.view(B, 1, 1, 1, 1) / M.view(B, 1, 1, 1, 1)
            # o = 0

            temp = y0 - transforms.fft2(x)
            temp[~mask, :] = 0
            z  = temp + o

        next_variables = torch.cat([x, z], dim=1)

        return next_variables


_solver_map = {
    'admm': ADMMSolver_CSMRI, 
    'hqs': HQSSolver_CSMRI,
    'pg': PGSolver_CSMRI,
    'apg': APGSolver_CSMRI,
    'redadmm': REDADMMSolver_CSMRI,
    'amp': AMPSolver_CSMRI
}

def create_solver_csmri(opt, denoiser):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser)
    else:
        raise NotImplementedError

    return solver
