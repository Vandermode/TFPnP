#!/usr/bin/env python3
import torch
import torch.utils.data.dataloader
from scipy.io import loadmat
from pathlib import Path

from env import CSMRIEnv
from dataset import CSMRIDataset, CSMRIEvalDataset
from solver import create_solver_csmri

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.trainer.mddpg.critic import ResNet_wobn
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelD
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    data_dir = Path('data')
    log_dir = Path(opt.output)
    mask_dir = data_dir / 'csmri' / 'masks'

    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']

    base_dim = CSMRIEnv.ob_base_dim
    actor = create_policy_network(opt, base_dim).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_csmri(opt, denoiser).to(device)
    num_var = solver.num_var

    # ---------------------------------------------------------------------------- #
    #                                     Valid                                    #
    # ---------------------------------------------------------------------------- #

    sigma_n_eval = 15
    val_roots = [data_dir / 'csmri' / 'Medical7_2020' / sampling_mask / str(sigma_n_eval) for sampling_mask in sampling_masks]
    val_datasets = [CSMRIEvalDataset(val_root, fns=['Brain']) for val_root in val_roots]

    val_loaders = [torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    val_names = [f'radial_128_2_{sigma_n_eval}', f'radial_128_4_{sigma_n_eval}', f'radial_128_8_{sigma_n_eval}']
    val_loaders = dict(zip(val_names, val_loaders))

    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)

    eval_env = CSMRIEnv(None, solver, max_episode_step=opt.max_episode_step, device=device)
    evaluator = Evaluator(opt, eval_env, val_loaders, device, log_dir / 'results')

    if opt.eval:
        actor_ckpt = torch.load(opt.resume)
        actor.load_state_dict(actor_ckpt)
        evaluator.eval(actor, step=opt.resume_step)
        return

    # ---------------------------------------------------------------------------- #
    #                                     Train                                    #
    # ---------------------------------------------------------------------------- #

    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    train_root = data_dir / 'Images_128'
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]
    train_dataset = CSMRIDataset(train_root, fns=None, masks=masks, noise_model=noise_model)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    env = CSMRIEnv(train_loader, solver, max_episode_step=opt.max_episode_step, device=device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 3e-4, 'actor': 1e-3}
        else:
            return {'critic': 1e-4, 'actor': 3e-4}

    critic = ResNet_wobn(base_dim+num_var, 18, 1).to(device)
    critic_target = ResNet_wobn(base_dim+num_var, 18, 1).to(device)

    trainer = MDDPGTrainer(opt, env, actor=actor,
                           critic=critic, critic_target=critic_target,
                           lr_scheduler=lr_scheduler, device=device,
                           log_dir=log_dir,
                           evaluator=evaluator, enable_tensorboard=True)
    if opt.resume:
        trainer.load_model(opt.resume, opt.resume_step)
    trainer.train()


if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)
