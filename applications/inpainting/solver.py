import torch
import numpy as np
from scipy.interpolate import griddata

from tfpnp.pnp.solver.base import PnPSolver
from tfpnp.pnp.denoiser import Denoiser

class InpaintingADMMPnPSolver(PnPSolver):
    def __init__(self, denoiser: Denoiser):
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
    
    def forward(self, inputs, parameters, iter_num=None):
        """
            img_L: [W, H, C], range = [0,1]
        """

        variables, (Stx, mask) = inputs
        rhos, sigmas = parameters
        
        x, v, u = torch.split(variables, variables.shape[1] // 3, dim=1)
        
        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = rhos.shape[-1]
            

        for i in range(iter_num):

            # --------------------------------
            # step 1, FFT
            # --------------------------------

            tau = rhos[:,i].float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
            xtilde = v - u
            rhs = Stx + tau * xtilde
            x = rhs / (mask + tau)

            # --------------------------------
            # step 2, denoiser
            # --------------------------------

            vtilde = x + u
            v = self.denoiser.denoise(vtilde, sigmas[:,i])


            # --------------------------------
            # step 3, u subproblem
            # --------------------------------

            u = u + x - v
        
        return torch.cat((x, v, u), dim=1)

    def get_output(self, state):
        # just return x after convert to real
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x
    
    def filter_additional_input(self, state):
        return (state['Stx'], state['mask'])
    
    def filter_hyperparameter(self, action):
        return action['mu'], action['sigma_d']
    
    
def Interpolation_OLRT(img, mask):
    """
    simulate random projection
    mask=0 denotes preserved pixels
    Delaunay triangulation based interpolation
    
    ```matlab
    [x,y,z] = ind2sub(size(c),find(c==0));
    [M,N,B]=size(im_n);
    [x1,y1,z1]=meshgrid(1:M,1:N,1:B);
    im_r=griddata(x,y,z,im_n(c==0),x1,y1,z1);
    for i =1:B
        im_r(:,:,i) = im_r(:,:,i)';
    end
    I=find(isnan(im_r)==0);
    I=find(isnan(im_r)==1);
    %J1=max(1,I-1);J2=min(M*N,I+1);
    im_r(I)=128;
    ```
    """
    w,h = img.shape
    idxs = np.argwhere(mask==1)
    x1, y1 = np.mgrid[0:w:1,0:h:1]
    img_r = griddata(idxs, img[mask==1], (x1.flatten(),y1.flatten()))
    img_r = img_r.reshape(w,h)
    img_r[np.isnan(img_r)] = 0.5
    return img_r

def Interpolation_OLRT_3D(img, mask):
    b = img.shape[-1]
    r = np.zeros_like(img)
    for i in range(b):
        r[:,:,i] = Interpolation_OLRT(img[:,:,i], mask[:,:,i])
    return r