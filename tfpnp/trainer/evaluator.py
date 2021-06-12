from tfpnp.util.metric import pnsr_qrnn3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from os.path import join
from scipy.io import savemat

from ..util.misc import AverageMeters, to_numpy, prRed

plt.switch_backend('agg')


class Evaluator(object):
    def __init__(self, opt, val_loaders, names, writer, keys=None, savedir=None):  
        self.val_loaders = val_loaders
        self.names = names
        self.max_step = opt.max_step
        self.env_batch = opt.env_batch
        self.writer = writer
        self.keys = keys or ('sigma_d',)
        self.savedir = savedir
        self.opt = opt

    def __call__(self, env, policy, step, loop_penalty=0):        
        # torch.manual_seed(1111)  # fix
        savedir = self.savedir
        keys = self.keys

        for name, val_loader in zip(self.names, self.val_loaders):
            avg_meters = AverageMeters()        
            observation = None
            for k, data in enumerate(val_loader):
                data_name = data['name'][0]
                data.pop('name')
                assert data['gt'].shape[0] == 1
                # reset at the start of episode                
                observation = env.reset(data=data)
                
                input, _, gt = env.get_images(observation)

                episode_steps = 0
                episode_reward = 0.
                assert observation is not None
                # start episode
                episode_reward = np.zeros(1)            
                all_done = False

                psnr_input = pnsr_qrnn3d(input, gt)
                sigma_d_seq = []
                mu_seq = []
                tau_seq = []
                psnr_seq = [psnr_input.item()]
                reward_seq = [0]

                data_name = str(k)

                if savedir is not None:
                    if not os.path.exists(join(savedir, name, data_name)):
                        os.makedirs(join(savedir, name, data_name))
                                           
                    Image.fromarray(input[0,...]).save(join(savedir, name, data_name, 'input.png'))
                    # Image.fromarray(input[0,...]).save(join(savedir, name, data_name, 'input_{:.2f}.png'.format(psnr_input.item())))
                    Image.fromarray(gt[0,...]).save(join(savedir, name, data_name, 'gt.png'))

                params_dict = {}
                
                ob = None
                while (episode_steps < self.max_step or not self.max_step):
                    action = policy(env.get_policy_state(observation), test=True)
                    
                    # for key in action:
                    #     if key not in params_dict:
                    #         params_dict[key] = []
                    #     params_dict[key].extend(to_numpy(action[key])[0])

                    ob, filtered_ob, reward, done, _ = env.step(action)
                    
                    if not done: 
                        reward = reward - loop_penalty

                    episode_reward += reward.item()
                    episode_steps += 1

                    # if 'sigma_d' in keys:
                    #     sigma_d_seq.extend(list(to_numpy(action['sigma_d'][0]*255)))
                    
                    # if 'mu' in keys:
                    #     mu_seq.extend(list(to_numpy(action['mu'][0])))
                    
                    # if 'tau' in keys:
                    #     tau_seq.extend(list(to_numpy(action['tau'][0])))

                    input, output, gt = env.get_images(ob)
                    cur_psnr = pnsr_qrnn3d(output, gt)
                    psnr_seq.append(cur_psnr.item())      
                    reward_seq.append(reward.item())

             
                    if done:
                        break

                input, output, gt = env.get_images(ob)                
                
                # if savedir is not None:
                #     savemat(join(savedir, name, '{}.mat'.format(data_name)), params_dict) # save learned parameters (figure 7)
                #     fig, _ = self.seq_plot(sigma_d_seq, 'Number of iterations (#IT.)', r'Denoising strength $\sigma$', 'blue')
                #     plt.savefig(join(savedir, name, data_name, 'sigma.pdf'), bbox_inches='tight')
                #     plt.clf()
                #     fig, _ = self.seq_plot(mu_seq, 'Number of iterations (#IT.)', r'Penalty parameter $\mu$', 'orange')
                #     plt.savefig(join(savedir, name, data_name, 'mu.pdf'), bbox_inches='tight')
                #     plt.clf()
                
                # if self.writer is not None:
                #     fig, _ = self.seq_plot(psnr_seq, '#IT.', 'PSNR')
                #     self.writer.add_figure('{}/lp{:.2f}/{}/psnr_seq'.format(name, loop_penalty, k), fig, step)

                #     if 'sigma_d' in keys:
                #         fig, _ = self.seq_plot(sigma_d_seq, '#IT.', r'$\sigma$')
                #         self.writer.add_figure('{}/lp{:.2f}/{}/sigma_d_seq'.format(name, loop_penalty, k), fig, step)
                    
                #     if 'mu' in keys:
                #         fig, _ = self.seq_plot(mu_seq, '#IT.', r'$\mu$')
                #         self.writer.add_figure('{}/lp{:.2f}/{}/mu_seq'.format(name, loop_penalty, k), fig, step)                    

                #     if 'tau' in keys:
                #         fig, _ = self.seq_plot(tau_seq, '#IT.', r'$\tau$')
                #         self.writer.add_figure('{}/lp{:.2f}/{}/tau_seq'.format(name, loop_penalty, k), fig, step)

                #     fig, ax = self.seq_plot(reward_seq, '#IT.', 'Reward')
                #     ax.hlines(y=0, xmin=0, xmax=len(reward_seq)-1, linestyles='dotted', colors='r')
                #     self.writer.add_figure('{}/lp{:.2f}/{}/reward_seq'.format(name, loop_penalty, k), fig, step)

                    # self.writer.add_image('{}/lp{:.2f}/{}/_gt.png'.format(name, loop_penalty, k), gt, step)
                    # self.writer.add_image('{}/lp{:.2f}/{}/_input.png'.format(name, loop_penalty, k), input, step)
                    # self.writer.add_image('{}/lp{:.2f}/{}/_output.png'.format(name, loop_penalty, k), output, step)

                psnr_finished = pnsr_qrnn3d(output, gt)

                avg_meters.update({'acc_reward': episode_reward, 'psnr': psnr_finished, 'iters': episode_steps})

            prRed('Step_{:07d}: {} | loop_penalty: {:.2f} | {}'.format(step - 1, name, loop_penalty, avg_meters))

            # if self.writer is not None:
            #     self.writer.add_scalar('validate/{}/lp{:.2f}/mean_acc_reward'.format(name, loop_penalty), avg_meters['acc_reward'], step)
            #     self.writer.add_scalar('validate/{}/lp{:.2f}/mean_psnr'.format(name, loop_penalty), avg_meters['psnr'], step)
            #     self.writer.add_scalar('validate/{}/lp{:.2f}/mean_iters'.format(name, loop_penalty), avg_meters['iters'], step)

            # return avg_meters

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
