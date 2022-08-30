#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat

from env import CSMRIEnv
from dataset import CSMRIDataset, CSMRIEvalDataset
from solver import create_solver_csmri

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelD
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']


def build_evaluator(data_dir, solver, sigma_n, save_dir):
    val_loaders = {}
    for sampling_mask in sampling_masks:
        root = data_dir / 'csmri' / 'Medical7_2020' / sampling_mask / str(sigma_n)
        dataset = CSMRIEvalDataset(root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        val_loaders[f'{sampling_mask}_{sigma_n}'] = loader

    eval_env = CSMRIEnv(None, solver, max_episode_step=opt.max_episode_step).to(device)
    evaluator = Evaluator(eval_env, val_loaders, save_dir)
    return evaluator


def train(opt, data_dir, mask_dir, policy, solver, log_dir):
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    train_root = data_dir / 'Images_128'
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]
    train_dataset = CSMRIDataset(train_root, fns=None, masks=masks, noise_model=noise_model)
    train_loader = DataLoader(train_dataset, opt.env_batch,
                              shuffle=True, num_workers=opt.num_workers,
                              pin_memory=True, drop_last=True)

    eval = build_evaluator(data_dir, solver, '15', log_dir / 'eval_results')
    
    env = CSMRIEnv(train_loader, solver, max_episode_step=opt.max_episode_step).to(device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 3e-4, 'actor': 1e-3}
        else:
            return {'critic': 1e-4, 'actor': 3e-4}

    trainer = MDDPGTrainer(opt, env, policy,
                           lr_scheduler=lr_scheduler, 
                           device=device,
                           log_dir=log_dir,
                           evaluator=eval, 
                           enable_tensorboard=True)
    if opt.resume:
        trainer.load_model(opt.resume, opt.resume_step)
    trainer.train()


def main(opt):
    data_dir = Path('data')
    log_dir = Path(opt.output)
    mask_dir = data_dir / 'csmri' / 'masks'

    base_dim = CSMRIEnv.ob_base_dim
    policy = create_policy_network(opt, base_dim).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_csmri(opt, denoiser).to(device)
    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)
        
    if opt.eval:
        ckpt = torch.load(opt.resume)
        policy.load_state_dict(ckpt)
        for sigma_n in [5, 10, 15]:
            save_dir = log_dir / 'test_results' / str(sigma_n)
            e = build_evaluator(data_dir, solver, sigma_n, save_dir)
            e.eval(policy, step=opt.resume_step)
            print('--------------------------')
        return 
    
    train(opt, data_dir, mask_dir, policy, solver, log_dir)

if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)
