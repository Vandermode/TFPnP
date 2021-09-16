import os
from tfpnp.pnp.solver.csmri import ADMMSolver_CSMRI
from tfpnp.pnp.denoiser import UNetDenoiser2D
import torch.utils.data
from scipy.io import loadmat

from tfpnp.data.dataset import CSMRIDataset, CSMRIEvalDataset, dict_to_device
from tfpnp.data.noise_models import GaussianModelD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # load data
    batch_size = 2
    
    train_dir = 'data/'
    mri_dir = 'data/'
    mask_dir = 'data/masks'
    
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']  # different masks
    train_root = os.path.join(train_dir, 'Medical_128')
    val_roots = [os.path.join(mri_dir, 'Medical7_2020', sampling_mask, '15') for sampling_mask in sampling_masks]
    obs_masks = [loadmat(os.path.join(mask_dir, '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks]
    
    train_dataset = CSMRIDataset(train_root, fns=None, masks=obs_masks, noise_model=noise_model, repeat=12*100)
    val_datasets = [CSMRIEvalDataset(val_root, fns=None) for val_root in val_roots]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    iterator = iter(train_loader)
    data = iterator.__next__()
    data = dict_to_device(data, device)
    
    # solve
    
    denoiser = UNetDenoiser2D('model/unet-nm.pt').to(device)
    solver = ADMMSolver_CSMRI(denoiser)
    
    solver_state = solver.reset(data)
    
    iter_num = 10
    rhos = torch.ones((batch_size, iter_num)).to(device)
    sigmas = torch.ones((batch_size, iter_num)).to(device)
    parameters = (rhos, sigmas)
    
    inputs = (solver_state, (data['y0'], data['mask']))
    states = solver.forward(inputs, parameters, iter_num)
    x, v, u = states
    
    print(x.shape)