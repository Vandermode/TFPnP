import torch
import numpy as np

from tfpnp.pnp.solver.base import ADMMSolver, HQSSolver, PGSolver, APGSolver, REDADMMSolver, AMPSolver
from tfpnp.utils import transforms


class SPIMixin:
    def filter_aux_inputs(self, state):
        return (state['x0'], state['K'])


class ADMMSolver_SPI(SPIMixin, ADMMSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def forward(self, inputs, parameters, iter_num=None):        
        # state: torch.cat([x, z, u], dim=1)
        # decode action
        variables, (x0, K) = inputs
        sigma_d, mu = parameters

        x, z, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        B = x.shape[0]
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = sigma_d.shape[-1]

        # K = K[0].item()
        # K1 = F.avg_pool2d(y0, K) * (K ** 2)
        K = K[:, 0, 0, 0].view(B, 1, 1, 1) * 10        
        K1 = x0 * (K ** 2)
        
        for i in range(iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i]        
            _mu = _mu.view(B, 1, 1, 1)

            # z step (x + u)
            z = transforms.spi_inverse(x + u, K1, K, _mu)

            # u step
            u = u + x - z

            # x step
            x = self.prox_mapping((z - u), _sigma_d)  # Plug-and-play proximal mapping            

        next_variables = torch.cat([x, z, u], dim=1)

        return next_variables


_solver_map = {
    'admm_spi': ADMMSolver_SPI, 
}

def create_solver_spi(opt, denoiser):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser)
    else:
        raise NotImplementedError

    return solver
