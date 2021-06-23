#!/usr/bin/env python3
from functools import partial
from tfpnp.util.metric import pnsr_qrnn3d
from tfpnp.trainer.a2cddpg.base import A2CDDPGTrainer
import cv2
import torch
from tensorboardX import SummaryWriter
from scipy.io import loadmat

from script.options import TrainOptions

from tfpnp.env.deblur import DeblurEnv
from tfpnp.data.dataset import HSIDeblurDataset
from tfpnp.pnp.solver.deblur import ADMMSolver_Deblur
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.policy.resnet import ResNetActor_HSI
from tfpnp.trainer.a2cddpg.critic import ResNet_wobn
from tfpnp.trainer.evaluator import Evaluator

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(opt):
    train_dataset = HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', target_size=(128,128))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.env_batch)
    
    val_datasets = [HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', training=False, target_size=(128,128))]
    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    val_names = ['icvl']
    
    return train_loader, val_loaders, val_names

def lr_scheduler(step):
    if step < 10000:
        lr = (3e-4, 1e-3)
    else:
        lr = (1e-4, 3e-4)
    return lr

if __name__ == "__main__":
    option = TrainOptions()
    opt = option.parse()

    opt.action_pack = 1
    
    writer = SummaryWriter('./train_log/{}'.format(opt.exp))

    train_loader, val_loaders, val_names = get_dataloaders(opt)

    policy_network = ResNetActor_HSI(189, opt.action_pack).to(device)
    critic = ResNet_wobn(189, 18, 1).to(device)
    critic_target = ResNet_wobn(189, 18, 1) .to(device)
    
    denoiser = GRUNetDenoiser('model/grunet-unet-qrnn3d.pth').to(device)
    solver = ADMMSolver_Deblur(denoiser)
    
    env = DeblurEnv(train_loader, solver, max_step=6)
    evaluator = Evaluator(opt, val_loaders, val_names, writer)
    
    trainer = A2CDDPGTrainer(opt, env, policy_network=policy_network, 
                             critic=critic, critic_target=critic_target, 
                             evaluator=evaluator, writer=writer)
    
    trainer.train()
