from tfpnp.util.metric import pnsr_qrnn3d
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.pnp.solver.deblur import ADMMSolver_Deblur
import torch.utils.data
import torch

from tfpnp.data.dataset import HSIDeblurDataset, dict_to_device
from tfpnp.pnp.util.dpir import utils_pnp as pnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    batch_size = 2
    
    train_dataset = HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    denoiser = GRUNetDenoiser('model/grunet-unet-qrnn3d.pth').to(device)
    solver = ADMMSolver_Deblur(denoiser)
    
    iterator = iter(train_loader)
    data = iterator.__next__()
    data = dict_to_device(data, device)
    
    solver_state = solver.reset(data)
    
    iter_num = 10
    rhos, sigmas = pnp.get_rho_sigma_admm(sigma=max(0.255/255., 0),
                                          iter_num=iter_num,
                                          modelSigma1=35, modelSigma2=10,
                                          w=1,
                                          lam=0.23)

    rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
    print(rhos.shape)   # torch.Size([10])
    print(sigmas.shape) # torch.Size([10])
    
    rhos = rhos.repeat(batch_size,1)
    sigmas = sigmas.repeat(batch_size,1)
    
    print(rhos.shape)   # torch.Size([10])
    print(sigmas.shape) # torch.Size([10])
    
    parameters = (rhos, sigmas)
    states = solver.forward(solver_state, parameters, iter_num)
    x, v, u = states
    
    print(x.shape)
    
    gt = data['gt'].detach().cpu().numpy()[0]
    x = x.detach().cpu().numpy()[0]
    
    psnr = pnsr_qrnn3d(gt, x)
    
    print(psnr)