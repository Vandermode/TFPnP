from tfpnp.util.metric import pnsr_qrnn3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from os.path import join
from scipy.io import savemat

from ..util.misc import AverageMeters, to_numpy, prRed
from ..pnp.util.dpir import utils_pnp as pnp

plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluatorDeblur(object):
    def __init__(self, opt, val_loaders, names, writer, psnr_fn=pnsr_qrnn3d, keys=None, savedir=None):  
        self.val_loaders = val_loaders
        self.names = names
        self.max_step = opt.max_step
        self.env_batch = opt.env_batch
        self.writer = writer
        self.keys = keys or ('sigma_d',)
        self.savedir = savedir
        self.opt = opt
        self.psnr_fn = psnr_fn

    def __call__(self, env, policy, step, loop_penalty=0):        
        # torch.manual_seed(1111)  # fix
        savedir = self.savedir
        keys = self.keys

        for name, val_loader in zip(self.names, self.val_loaders):
            avg_meters = AverageMeters()        
            observation = None
            for k, data in enumerate(val_loader):
                if name in data.keys():
                    data_name = data['name'][0]
                    data.pop('name')
                else:
                    data_name = 'test'
                assert data['gt'].shape[0] == 1
                # reset at the start of episode                
                observation = env.reset(data=data)
                
                input, _, gt = env.get_images(observation)

                episode_steps = 0
                episode_reward = 0.
                # start episode
                episode_reward = np.zeros(1)            

                psnr_input = self.psnr_fn(input, gt)
                psnr_seq = [psnr_input.item()]
                reward_seq = [0]

                data_name = str(k)

                if savedir is not None:
                    if not os.path.exists(join(savedir, name, data_name)):
                        os.makedirs(join(savedir, name, data_name))
                                           
                    Image.fromarray(input[0,...]).save(join(savedir, name, data_name, 'input.png'))
                    # Image.fromarray(input[0,...]).save(join(savedir, name, data_name, 'input_{:.2f}.png'.format(psnr_input.item())))
                    Image.fromarray(gt[0,...]).save(join(savedir, name, data_name, 'gt.png'))

                ob = None
                while (episode_steps < self.max_step or not self.max_step):
                    action = policy(env.get_policy_state(observation), test=True)
    
                    ob, filtered_ob, reward, done, _ = env.step(action)
                    
                    if not done: 
                        reward = reward - loop_penalty

                    episode_reward += reward.item()
                    episode_steps += 1

                    input, output, gt = env.get_images(ob)
                    cur_psnr = self.psnr_fn(output, gt)
                    psnr_seq.append(cur_psnr.item())      
                    reward_seq.append(reward.item())
  
                    if done:
                        break

                input, output, gt = env.get_images(ob)                

                psnr_finished = self.psnr_fn(output, gt)

                # fix schema
                observation = env.reset(data=data)
                solver_state = env.state['solver']
                
                iter_num = self.max_step
                rhos, sigmas = pnp.get_rho_sigma_admm(sigma=max(0.255/255., 0),
                                                    iter_num=iter_num,
                                                    modelSigma1=35, modelSigma2=10,
                                                    w=1,
                                                    lam=0.23)

                rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
                
                
                batch_size = observation.shape[0]
                rhos = rhos.repeat(batch_size,1)
                sigmas = sigmas.repeat(batch_size,1)
                
                parameters = (rhos, sigmas)
                inputs = (solver_state, (env.state['FB'], env.state['FBC'], env.state['F2B'], env.state['FBFy']))
                states = env.solver.forward(inputs, parameters, iter_num)
                
                x = _pre_img(env.solver.get_output(states))
                psnr_fixed = self.psnr_fn(x, gt)
                
                avg_meters.update({'acc_reward': episode_reward, 'psnr': psnr_finished, 'psnr_fixed': psnr_fixed, 'iters': episode_steps})

            prRed('Step_{:07d}: {} | loop_penalty: {:.2f} | {}'.format(step - 1, name, loop_penalty, avg_meters))

    def seq_plot(self, seq, xlabel, ylabel, color='blue'):
        # fig, ax = plt.subplots(1, 1)
        # fig, ax = plt.subplots(1, 1, figsize=(6,4))
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        # ax.plot(np.array(seq))
        ax.plot(np.arange(1, len(seq)+1), np.array(seq), 'o--', markersize=10, linewidth=2, color=color)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        # plt.xticks(fontsize=16)
        
        xticks = list(range(1, len(seq)+1, max(len(seq)//5,1)))
        if xticks[-1] != len(seq):
            xticks.append(len(seq))

        plt.xticks(xticks, fontsize=16)
        
        return fig, ax
    
def _pre_img(img):
    img = to_numpy(img[0,...])
    img = np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8), 3, axis=0)
    return img