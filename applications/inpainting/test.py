from tfpnp.utils.misc import torch2img255
import torch
import torch.utils.data
import numpy as np
import os
from os.path import join
from functools import partial
from skimage.metrics.simple_metrics import peak_signal_noise_ratio

from dataset import HSIInpaintingDataset
from env import HSIInpaintingEnv
from solver import InpaintingADMMPnPSolver
import utils.dpir.utils_pnp as pnp

from tfpnp.utils.metric import psnr_qrnn3d
from tfpnp.policy.resnet import ResNetActor_HSI
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.eval import evaluator
from tfpnp.env import PnPEnv
from tfpnp.utils.visualize import save_img, seq_plot

class Evaluator:
    def __init__(self, policy_network, env:PnPEnv, max_step, savedir=None):
        self.policy_network = policy_network
        self.env = env
        self.max_step = max_step
        self.psnr_fn = partial(peak_signal_noise_ratio, data_range=255)
        self.savedir = savedir
        
        self.policy_network.eval()
    
    def eval(self, data):
        observation = self.env.reset(data=data)
                
        input, _, gt = self.env.get_images(observation)
        
        episode_steps = 0
        episode_reward = np.zeros(1)     

        psnr_input = self.psnr_fn(input[0], gt[0])
        psnr_seq = [psnr_input.item()]
        reward_seq = [0]
        
        mu_seq = []
        sigma_seq = []
        
        ob = observation
        while episode_steps < self.max_step :
            policy_state = self.env.get_policy_state(ob)
            action = self.select_action(policy_state, test=True)

            ob, filtered_ob, reward, done, _ = self.env.step(action)
            
            episode_reward += reward.item()
            episode_steps += 1

            input, output, gt = self.env.get_images(ob)
            cur_psnr = self.psnr_fn(output[0], gt[0])
            psnr_seq.append(cur_psnr.item())      
            reward_seq.append(reward.item())

            mu_seq.append(action['mu'].item())
            sigma_seq.append(action['sigma_d'].item() * 255)
        
            if done:
                break
        
        input, output, gt = self.env.get_images(ob)               
        psnr_finished = self.psnr_fn(output[0], gt[0])
        psnr_fixed, psnr_best, psnr_seq_ours, mu_ours_seq, sigma_ours_seq = self.eval_fixed(data, self.env, self.max_step)
        print('name{}, step:{}, psnr - input:{:.2f}, tfpnp:{:.2f}, fixed:{:.2f}, fixed(best): {:.2f}'.format(data['name'], episode_steps, psnr_input, psnr_finished, psnr_fixed, psnr_best))

        # save imgs
        if self.savedir is not None:
            base_dir = join(self.savedir, 'test', data['name'][0])
            os.makedirs(base_dir, exist_ok=True)
        
            seq_plot(psnr_seq, 'step', 'psnr', save_path=join(base_dir, 'psnr.png'))     
            seq_plot(psnr_seq_ours, 'step', 'psnr_ours', save_path=join(base_dir, 'psnr_ours.png'))     
            seq_plot(mu_seq, 'step', 'mu', save_path=join(base_dir, 'mu.png'))     
            seq_plot(sigma_seq, 'step', 'sigma', save_path=join(base_dir, 'sigma.png')) 
            seq_plot(mu_ours_seq, 'step', 'mu_ours', save_path=join(base_dir, 'mu_ours.png'))     
            seq_plot(sigma_ours_seq, 'step', 'sigma_ours', save_path=join(base_dir, 'sigma_ours.png'))     
        
        return psnr_input, psnr_finished, psnr_fixed, psnr_best

    def select_action(self, state, idx_stop=None, test=False):
        with torch.no_grad():
            action, _, _ = self.policy_network(state, idx_stop, not test)
        return action
    
    def eval_fixed(self, data, env:PnPEnv, iter_num):
        observation = env.reset(data=data)
        input, _, gt = env.get_images(observation)
        solver_state = env.state['solver']
        
        rhos, sigmas = pnp.get_rho_sigma_admm(sigma=max(0.255/255., 0),
                                            iter_num=iter_num,
                                            modelSigma1=45, modelSigma2=10,
                                            w=1,
                                            lam=0.23)

        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
        
        batch_size = observation.shape[0]
        rhos = rhos.repeat(batch_size,1)
        sigmas = sigmas.repeat(batch_size,1)

        psnr_seq = []
        psnr_best = self.psnr_fn(input[0], gt[0])
        for i in range(iter_num):
            parameters = (rhos[:,i:i+1], sigmas[:,i:i+1])
            inputs = (solver_state, (env.state['Stx'], env.state['mask']))
            solver_state = env.solver.forward(inputs, parameters, 1)
            
            x = torch2img255(env.solver.get_output(solver_state))
            psnr_fixed = self.psnr_fn(x[0], gt[0])
            psnr_seq.append(psnr_fixed)
            if psnr_fixed > psnr_best:
                psnr_best = psnr_fixed
            
        return psnr_fixed, psnr_best, psnr_seq, rhos.detach().cpu().numpy().tolist()[0], (sigmas*255).detach().cpu().numpy().tolist()[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = HSIInpaintingDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', training=False, target_size=(128,128))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)
    
    policy_network = ResNetActor_HSI(187, 1).to(device)
    policy_network.load_state_dict(torch.load('checkpoints/sigma0/actor_0000199.pkl'))
    
    denoiser = GRUNetDenoiser().to(device)
    solver = InpaintingADMMPnPSolver(denoiser)
    
    iter_num = 40
    env = HSIInpaintingEnv(None, solver, max_step=iter_num, device=device)
    
    evaluator = Evaluator(policy_network, env, iter_num, savedir='log/baseline/')
    
    psnr_inputs, psnr_finisheds, psnr_fixeds, psnr_bests = [], [], [], []
    for data in val_loader:
        psnr_input, psnr_finished, psnr_fixed, psnr_best = evaluator.eval(data)
        psnr_inputs.append(psnr_input)
        psnr_finisheds.append(psnr_finished)
        psnr_fixeds.append(psnr_fixed)
        psnr_bests.append(psnr_best)
    print('avg pnsr input: ', sum(psnr_inputs) / len(psnr_inputs))    
    print('avg pnsr tfpnp: ', sum(psnr_finisheds) / len(psnr_finisheds))    
    print('avg pnsr fixed: ', sum(psnr_fixeds) / len(psnr_fixeds))    
    print('avg pnsr fixed(best): ', sum(psnr_bests) / len(psnr_bests))    
    