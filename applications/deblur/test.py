from tfpnp.utils.misc import torch2img255
import torch
import torch.utils.data
import os
from os.path import join

from dataset import HSIDeblurDataset
from env import DeblurEnv
from solver import ADMMSolver_Deblur
import utils.dpir.utils_pnp as pnp

from tfpnp.utils.metric import psnr_qrnn3d
from tfpnp.policy.resnet import ResNetActor_HSI
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.trainer import evaluator
from tfpnp.env import PnPEnv
from tfpnp.utils.visualize import seq_plot
from tfpnp.eval.evaluator import eval_single


class Evaluator:
    def __init__(self, policy_network, env:PnPEnv, savedir=None):
        self.policy_network = policy_network
        self.env = env
        self.max_step = 10
        self.psnr_fn = psnr_qrnn3d
        self.savedir = savedir
        
        self.policy_network.eval()
    
    def eval(self, data):
        psnr_input, psnr_finished, info, imgs = eval_single(self.env, data, self.select_action, 
                                                            max_step=self.max_step,
                                                            loop_penalty=0.05,
                                                            metric=self.psnr_fn)

        episode_steps, episode_reward, psnr_seq, reward_seq, action_seqs = info
        # input, output_init, output, gt = imgs
        psnr_fixed, psnr_best, psnr_seq_ours, mu_ours_seq, sigma_ours_seq = self.eval_fixed(data, self.env, self.max_step)
        
        print('name{}, step:{}, psnr - input:{:.2f}, tfpnp:{:.2f}, fixed:{:.2f}, fixed(best): {:.2f}'.format(data['name'], episode_steps, psnr_input, psnr_finished, psnr_fixed, psnr_best))

        # save imgs
        if self.savedir is not None:
            base_dir = join(self.savedir, 'test', data['name'][0])
            os.makedirs(base_dir, exist_ok=True)
        
            seq_plot(psnr_seq, 'step', 'psnr', save_path=join(base_dir, 'psnr.png'))     
            seq_plot(psnr_seq_ours, 'step', 'psnr_ours', save_path=join(base_dir, 'psnr_ours.png'))     
            for k, v in action_seqs.items():
                seq_plot(v, 'step', k, save_path=join(base_dir, k+'.png'))
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
                                            modelSigma1=35, modelSigma2=10,
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
            inputs = (solver_state, (env.state['FB'], env.state['FBC'], env.state['F2B'], env.state['FBFy']))
            solver_state = env.solver.forward(inputs, parameters, 1)
            
            x = torch2img255(env.solver.get_output(solver_state))
            psnr_fixed = self.psnr_fn(x[0], gt[0])
            psnr_seq.append(psnr_fixed)
            if psnr_fixed > psnr_best:
                psnr_best = psnr_fixed
            
        return psnr_fixed, psnr_best, psnr_seq, rhos.detach().cpu().numpy().tolist()[0], (sigmas*255).detach().cpu().numpy().tolist()[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/cave_512_15', training=False, target_size=(128,128))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)
    
    policy_network = ResNetActor_HSI(189, 1).to(device)
    policy_network.load_state_dict(torch.load('checkpoints/maxstep6/actor_0000599.pkl'))
    
    denoiser = GRUNetDenoiser().to(device)
    solver = ADMMSolver_Deblur(denoiser)
    
    iter_num = 15
    env = DeblurEnv(None, solver, max_step=iter_num, device=device)
    
    evaluator = Evaluator(policy_network, env, savedir='log/baseline/')
    
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
    