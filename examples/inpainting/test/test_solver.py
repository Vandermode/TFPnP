import torch
import torch.utils.data
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
import sys
sys.path += ['/home/laizeqiang/Desktop/lzq/projects/tfpnp/tfpnp2/','/home/laizeqiang/Desktop/lzq/projects/tfpnp/tfpnp2/examples/inpainting']

from dataset import HSIInpaintingDataset
from solver import InpaintingADMMPnPSolver
from utils.dpir import utils_pnp as pnp

from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.data.util import dict_to_device

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    batch_size = 2
    train_dataset = HSIInpaintingDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', target_size=(128,128), training=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    denoiser = GRUNetDenoiser().to(device)
    solver = InpaintingADMMPnPSolver(denoiser)
    
    for data in train_loader:
        data = dict_to_device(data, device)
        
        solver_state = solver.reset(data)
        print(solver_state.shape)
        
        iter_num = 10
        rhos, sigmas = pnp.get_rho_sigma_admm(sigma=max(0.255/255., 0),
                                            iter_num=iter_num,
                                            modelSigma1=35, modelSigma2=10,
                                            w=1,
                                            lam=0.23)

        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        rhos = rhos.repeat(batch_size,1)
        sigmas = sigmas.repeat(batch_size,1)
        
        print(rhos.shape)   # torch.Size([10])
        print(sigmas.shape) # torch.Size([10])
        
        parameters = (rhos, sigmas)
        inputs = (solver_state, (data['Stx'], data['mask']))
        states = solver.forward(inputs, parameters, iter_num)
        
        print(states.shape) # [2,31,512,512] 
        
        x = solver.get_output(states)
        
        print(x.shape)
        
        gt = data['gt'].detach().cpu().numpy()[0]
        x = x.detach().cpu().numpy()[0]
        
        psnr = peak_signal_noise_ratio(gt, x, data_range=1)
        
        print(psnr)