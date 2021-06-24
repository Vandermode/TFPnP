import torch
import torch.utils.data
import numpy as np

from dataset import HSIDeblurDataset
from env import DeblurEnv
from solver import ADMMSolver_Deblur

from tfpnp.utils.metric import pnsr_qrnn3d
from tfpnp.policy.resnet import ResNetActor_HSI
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.trainer import evaluator


class Evaluator:
    def __init__(self, policy_network, env):
        self.policy_network = policy_network
        self.env = env
        self.max_step = 10
        self.psnr_fn = pnsr_qrnn3d
        
        self.policy_network.eval()
    
    def eval(self, data):
        observation = self.env.reset(data=data)
                
        input, _, gt = self.env.get_images(observation)
        episode_steps = 0
        episode_reward = np.zeros(1)     

        psnr_input = self.psnr_fn(input, gt)
        psnr_seq = [psnr_input.item()]
        reward_seq = [0]
        
        while episode_steps < self.max_step :
            action = self.select_action(self.env.get_policy_state(observation), test=True)

            ob, filtered_ob, reward, done, _ = self.env.step(action)

            episode_reward += reward.item()
            episode_steps += 1

            input, output, gt = self.env.get_images(ob)
            cur_psnr = self.psnr_fn(output, gt)
            psnr_seq.append(cur_psnr.item())      
            reward_seq.append(reward.item())

            if done:
                break
        
        input, output, gt = self.env.get_images(ob)               
        psnr_finished = self.psnr_fn(output, gt)
        print(episode_steps, psnr_input, psnr_finished)
        

    def select_action(self, state, idx_stop=None, test=False):
        with torch.no_grad():
            action, _, _ = self.policy_network(state, idx_stop, not test)
        return action
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', training=False, target_size=(128,128))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)
    
    policy_network = ResNetActor_HSI(189, 1).to(device)
    policy_network.load_state_dict(torch.load('checkpoints/baseline/actor_0000299.pkl'))
    
    denoiser = GRUNetDenoiser().to(device)
    solver = ADMMSolver_Deblur(denoiser)
    
    env = DeblurEnv(None, solver, max_step=15, device=device)
    
    evaluator = Evaluator(policy_network, env)
    
    for data in val_loader:
        evaluator.eval(data)