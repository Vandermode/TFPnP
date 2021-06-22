from collections import namedtuple
from tfpnp.pnp.util.transforms import complex2channel, complex2real

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tensorboardX.writer import SummaryWriter

from ...util.misc import prRed, prBlack, soft_update, hard_update
from ...env.base import PnPEnv
from ...util.rpm import ReplayMemory

"""[summary]
https://www.jianshu.com/p/f9e7140ce19d
"""


def lr_scheduler(step):
    if step < 10000:
        return {'critic': 3e-4, 'actor': 1e-3}
    else:
        return {'critic': 1e-4, 'actor': 3e-4}

class A2CDDPGTrainer:
    def __init__(self, opt, env: PnPEnv, policy_network, critic, critic_target,
                 evaluator=None, writer:SummaryWriter=None):
        self.opt = opt
        self.env = env
        self.actor = policy_network
        self.critic = critic
        self.critic_target = critic_target
        self.evaluator = evaluator
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.total_steps = opt.epochs * opt.steps_per_epoch

        self.buffer = ReplayMemory(opt.rmsize * opt.max_step)
        
        self.optimizer_actor = Adam(self.actor.parameters())
        self.optimizer_critic = Adam(self.critic.parameters())
        
        self.criterion = nn.MSELoss()   # criterion for value loss
        
        hard_update(self.critic_target, self.critic)
    
    def select_action(self, state, idx_stop=None, test=False):
        self.actor.eval()
        with torch.no_grad():
            action, _, _ = self.actor(state, idx_stop, not test)
        self.actor.train()
        return action
    
    def save_experience(self, ob):
        ob_detached = ob.clone().detach().cpu() # crucial to prevent out of gpu memory
        for i in range(ob_detached.shape[0]):
            self.buffer.store(ob_detached[i])
    
    def train(self):
        # get initial observation
        ob = self.env.reset()
        episode, episode_step = 0, 0
        epoch = 0
        
        for step in range(self.total_steps):
            # select a action
            # TODO: 1. sample from action space at the first few steps for better exploration. 2. Noise action
            action = self.select_action(self.env.get_policy_state(ob))
            
            # step the env
            _, ob2_filtered, _, done, _ = self.env.step(action)
            episode_step += 1
            
            # store experience to replay buffer: in a2cddpg, we only need ob actually
            self.save_experience(ob)
            
            ob = ob2_filtered
            
            # end of trajectory handling
            if done or (episode_step == self.opt.max_step):
                if self.evaluator is not None and (episode+1) % self.opt.eval_per_episode == 0:
                    self.evaluator(self.env, self.select_action, step, self.opt.loop_penalty)
                
                if step > self.opt.warmup:
                    self.updaet_policy(episode, step)
                
                ob = self.env.reset()
                # from hdf5storage import savemat
                # savemat('state.mat', {'ob': ob.detach().cpu().numpy()})
                episode += 1
                episode_step = 0

            # end of epoch handling
            if (step + 1) % self.opt.steps_per_epoch == 0:
                epoch = (step+1) // self.opt.steps_per_epoch
                epoch += 1

                if (epoch % self.opt.save_freq == 0) or (epoch == self.opt.epochs):
                    prRed('Saving model at Step_{:07d}...'.format(step))
                    self.save_model(self.opt.save_path, step)
    
    
    def updaet_policy(self, episode, step):
        tot_Q, tot_value_loss, tot_dist_entropy = 0, 0, 0
        lr = lr_scheduler(step)
        
        for _ in range(self.opt.episode_train_times):
            import random
            random.seed(2021)
            samples = self.buffer.sample_batch(self.opt.env_batch)
            Q, value_loss, dist_entropy = self.update(samples=samples, lr=lr)
            
            tot_Q += Q
            tot_value_loss += value_loss
            tot_dist_entropy += dist_entropy
        
        mean_Q = tot_Q / self.opt.episode_train_times
        mean_dist_entropy = tot_dist_entropy / self.opt.episode_train_times
        mean_value_loss = tot_value_loss / self.opt.episode_train_times
        
        if self.writer is not None:
            self.writer.add_scalar('train/critic_lr', lr['critic'], step)
            self.writer.add_scalar('train/actor_lr', lr['actor'], step)
            self.writer.add_scalar('train/Q', mean_Q, step)
            self.writer.add_scalar('train/dist_entropy', mean_dist_entropy, step)
            self.writer.add_scalar('train/critic_loss', mean_value_loss, step)

        prBlack('#{}: steps: {} | Q: {:.2f} | dist_entropy: {:.2f} | critic_loss: {:.2f}' \
            .format(episode, step, mean_Q, mean_dist_entropy, mean_value_loss))
    
    
    def update(self, samples, lr:dict):
        # update learning rate
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = lr['actor']
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = lr['critic']
            
        # convert list of named tuple into named tuple of batch
        state = self.convert2batch(samples)        

        from hdf5storage import savemat
        
        
        num_var = 3
        gt = state[:, 0:1, ...]
        variables = state[:, 1:num_var+1, ...]
        y0 = state[:, num_var+1:num_var+2, ...]
        ATy0 = state[:, num_var+2:num_var+3, ...]
        mask = state[:, num_var+3:num_var+4, ...]
        T = state[:, num_var+4:num_var+5, ...]
        sigma_n = state[:, num_var+5:num_var+6, ...]
        
        policy_state = self.env.get_policy_state(state)
        
        eval_state = policy_state

        savemat('ob.mat', {'state':eval_state.detach().cpu().numpy()})


        action, action_log_prob, dist_entropy = self.actor(policy_state)
        
        _mask = complex2real(mask).bool() 
        inputs = (variables, (y0, _mask))
        output0 = complex2real(variables[:, 0:1,...])
        gt_real = complex2real(gt)
        variables2, reward = self.env.forward(inputs, output0, gt_real, action)
        
        reward -= self.opt.loop_penalty
        
        eval_state2 = torch.cat([complex2real(variables2), 
                                  complex2channel(y0),
                                  complex2real(ATy0),
                                  complex2real(mask), 
                                  complex2real(T) + 1/self.env.max_step, 
                                  complex2real(sigma_n)], 1)
        
        # compute actor critic loss for discrete action
        V_cur = self.critic(eval_state)
        with torch.no_grad():
            V_next_target = self.critic_target(eval_state2)
            V_next_target = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next_target
            Q_target = V_next_target + reward 
        advantage = (Q_target - V_cur).clone().detach()
        a2c_loss = action_log_prob * advantage
        
        # compute ddpg loss for continuous actions
        V_next = self.critic(eval_state2)
        V_next = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next
        ddpg_loss = V_next + reward
        
        entroy_regularization = dist_entropy
        
        policy_loss = - (a2c_loss + ddpg_loss + self.opt.lambda_e * entroy_regularization).mean()
        value_loss = self.criterion(Q_target, V_cur)
        
        # perform one step gradient descent
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.optimizer_actor.step()
        
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.optimizer_critic.step()
        
        # soft update target network
        soft_update(self.critic_target, self.critic, self.opt.tau)
        
        return -policy_loss.item(), value_loss.item(), entroy_regularization.mean().item()

    def convert2batch(self, states):
        return torch.stack(states, dim=0).to(self.device)
        
    def save_model(self):
        pass
    
    def load_model(self):
        pass
