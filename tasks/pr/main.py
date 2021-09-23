#!/usr/bin/env python3
import numpy as np
import torch
from tensorboardX import SummaryWriter
from scipy.io import loadmat
from pathlib import Path

from env import PREnv
from dataset import PRDataset
from solver import create_solver_pr

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.trainer.mddpg.critic import ResNet_wobn
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import PoissonModel
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    data_dir = Path('data')
    log_dir = Path('log') / opt.exp
    mask_dir = data_dir / 'masks'    
    
    writer = SummaryWriter(log_dir)

    alphas = [9, 27, 81]
    noise_model = PoissonModel(alphas)

    meta_info = loadmat(mask_dir / 'pr_x4.mat')
    obs_mask = meta_info.get('mask')
    obs_mask = np.stack((obs_mask.real, obs_mask.imag), axis=-1)
    
    train_root = data_dir / 'Images_128'
    val_root = data_dir / 'PrDeep_12'

    train_dataset = PRDataset(train_root, fns=None, masks=[obs_mask], noise_model=noise_model)
    val_datasets = [PRDataset(val_root, fns=None, masks=[obs_mask], noise_model=PoissonModel([alpha])) for alpha in alphas]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]    

    val_names = [f'alpha_{alpha}' for alpha in alphas]

    val_loaders = dict(zip(val_names, val_loaders))

    base_dim = 14
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_pr(opt, denoiser).to(device)
    actor = create_policy_network(opt, base_dim).to(device)  # policy network
    num_var = solver.num_var
    
    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)

    env = PREnv(train_loader, solver, max_episode_step=opt.max_episode_step, device=device)
    eval_env = PREnv(None, solver, max_episode_step=opt.max_episode_step, device=device)
    evaluator = Evaluator(opt, eval_env, val_loaders, writer, device)
    
    if opt.eval:
        actor_ckpt = torch.load(opt.resume)
        actor.load_state_dict(actor_ckpt)
        evaluator.eval(actor, step=opt.resume_step)
        return

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 1e-4, 'actor': 5e-5}
        else:
            return {'critic': 5e-5, 'actor': 1e-5}

    critic = ResNet_wobn(base_dim+num_var, 18, 1).to(device)
    critic_target = ResNet_wobn(base_dim+num_var, 18, 1).to(device)        

    trainer = MDDPGTrainer(opt, env, actor=actor,
                             critic=critic, critic_target=critic_target, 
                             lr_scheduler=lr_scheduler, device=device, 
                             evaluator=evaluator, writer=writer)
    trainer.train()


if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)
