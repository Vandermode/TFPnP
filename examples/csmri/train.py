#!/usr/bin/env python3
import os
import torch
from tensorboardX import SummaryWriter
from scipy.io import loadmat

from options import TrainOptions
from env import CSMRIEnv
from dataset import CSMRIDataset, CSMRIEvalDataset
from noise_models import GaussianModelD

from tfpnp.pnp.solver.csmri import ADMMSolver_CSMRI
from tfpnp.pnp.denoiser import UNetDenoiser2D
from tfpnp.policy.resnet import ResNetActor
from tfpnp.trainer import A2CDDPGTrainer
from tfpnp.trainer.a2cddpg.critic import ResNet_wobn
from tfpnp.trainer.evaluator import Evaluator

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(train_dir, mri_dir, mask_dir, opt):
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']  # different masks
    train_root = os.path.join(train_dir, 'Medical_128')
    val_roots = [os.path.join(mri_dir, 'Medical7_2020', sampling_mask, '15') for sampling_mask in sampling_masks]
    obs_masks = [loadmat(os.path.join(mask_dir, '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks]
    
    train_dataset = CSMRIDataset(train_root, fns=None, masks=obs_masks, noise_model=noise_model, repeat=12*100)
    val_datasets = [CSMRIEvalDataset(val_root, fns=None) for val_root in val_roots]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    val_names = ['m7x2', 'm7x4', 'm7x8']
    
    return train_loader, val_loaders, val_names

def lr_scheduler(step):
    if step < 10000:
        lr = (3e-4, 1e-3)
    else:
        lr = (1e-4, 3e-4)
    return lr

if __name__ == "__main__":
    train_dir = 'data'
    mask_dir = 'data/masks'
    mri_dir = 'data/'
        
    option = TrainOptions()
    opt = option.parse()

    writer = SummaryWriter('./train_log/{}'.format(opt.exp))

    train_loader, val_loaders, val_names = get_dataloaders(train_dir, mri_dir, mask_dir, opt)

    policy_network = ResNetActor(6+3, opt.action_pack).to(device)
    critic = ResNet_wobn(9, 18, 1).to(device)
    critic_target = ResNet_wobn(9, 18, 1) .to(device)
    
    denoiser = UNetDenoiser2D().to(device)
    solver = ADMMSolver_CSMRI(denoiser)
    env = CSMRIEnv(train_loader, solver, max_step=opt.max_step, device=device)
    evaluator = Evaluator(opt, val_loaders, val_names, writer)
    
    trainer = A2CDDPGTrainer(opt, env, policy_network=policy_network, 
                             critic=critic, critic_target=critic_target, 
                             device=device, evaluator=evaluator, writer=writer)
    
    trainer.train()
