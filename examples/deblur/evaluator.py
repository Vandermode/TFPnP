import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from os.path import join

from utils.dpir import utils_pnp as pnp
from tfpnp.utils.metric import pnsr_qrnn3d
from tfpnp.utils.misc import MetricTracker, prRed, torch2img255
from tfpnp.env.base import PnPEnv

plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluatorDeblur(object):
    def __init__(self, opt, env:PnPEnv, val_loaders, writer, savedir=None):  
        self.opt = opt
        self.env = env
        self.val_loaders = val_loaders
        self.writer = writer
        self.savedir = savedir

    def eval(self, policy, step):        
        for name, val_loader in self.val_loaders.items():
            metric_tracker = MetricTracker()        
            for index, data in enumerate(val_loader):
                if name in data.keys():
                    data_name = data['name'][0]
                    data.pop('name')
                else:
                    data_name = 'case' + str(index)
                    
                assert data['gt'].shape[0] == 1
                
                # reset at the start of episode                
                observation = self.env.reset(data=data)
                input, _, gt = self.env.get_images(observation)
                psnr_input = pnsr_qrnn3d(input, gt)
   
                if self.savedir is not None:
                    if not os.path.exists(join(self.savedir, name, data_name)):
                        os.makedirs(join(self.savedir, name, data_name))
                                           
                    Image.fromarray(input[0,...]).save(join(self.savedir, name, data_name, 'input.png'))
                    Image.fromarray(gt[0,...]).save(join(self.savedir, name, data_name, 'gt.png'))
                
                psnr_finished, episode_steps, episode_reward, psnr_seq, reward_seq = self._eval_policy_network(policy, data)
                psnr_fixed = self._eval_log_descent(data)
                
                metric_tracker.update({'acc_reward': episode_reward, 'psnr': psnr_finished, 'psnr_fixed': psnr_fixed, 'iters': episode_steps})

                
            prRed('Step_{:07d}: {} | {}'.format(step - 1, name, metric_tracker))


    def _eval_policy_network(self, policy, data):
        observation = self.env.reset(data=data)
        
        episode_steps = 0
        episode_reward = np.zeros(1)            

        psnr_seq = []
        reward_seq = [0]
        
        ob = observation
        while (episode_steps < self.opt.max_step or not self.opt.max_step):
            action = policy(self.env.get_policy_state(ob), test=True)
            
            # since batch size = 1, ob and ob_masked are always identicial
            ob, _, reward, done, _ = self.env.step(action) 
            
            if not done: 
                reward = reward - self.opt.loop_penalty

            episode_reward += reward.item()
            episode_steps += 1

            _, output, gt = self.env.get_images(ob)
            cur_psnr = pnsr_qrnn3d(output, gt)
            psnr_seq.append(cur_psnr.item())      
            reward_seq.append(reward.item())

            if done:
                break

        _, output, gt = self.env.get_images(ob)                

        psnr_finished = pnsr_qrnn3d(output, gt)

        return psnr_finished, episode_steps, episode_reward, psnr_seq, reward_seq
    
    def _eval_log_descent(self, data):
        observation = self.env.reset(data=data)
        _, _, gt = self.env.get_images(observation)         
        
        solver_state = self.env.state['solver']
        
        iter_num = self.opt.max_step * self.opt.action_pack
        rhos, sigmas = pnp.get_rho_sigma_admm(sigma=max(0.255/255., 0),
                                            iter_num=iter_num,
                                            modelSigma1=35, modelSigma2=10,
                                            w=1,
                                            lam=0.23)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
        
        
        batch_size = observation.shape[0]
        rhos = rhos.repeat(batch_size,1)
        sigmas = sigmas.repeat(batch_size,1)
        idx_stop = torch.ones (batch_size).to(device)
        
        ob2,_,_,_,_ = self.env.step({'mu': rhos, 'sigma_d': sigmas, 'idx_stop': idx_stop})
        
        # parameters = (rhos, sigmas)
        # inputs = (solver_state, self.env.solver.filter_additional_input(self.env.state))
        # states = self.env.solver.forward(inputs, parameters, iter_num)
        
        # x = torch2img255(self.env.solver.get_output(states))
        
        _, output, gt = self.env.get_images(ob2)
        
        psnr_fixed = pnsr_qrnn3d(output, gt)
        
        return psnr_fixed