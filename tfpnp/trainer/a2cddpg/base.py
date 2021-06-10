from collections import namedtuple

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
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        self.optimizer_actor = Adam(self.actor.parameters())
        self.optimizer_critic = Adam(self.critic.parameters())
        
        self.criterion = nn.MSELoss()   # criterion for value loss
        
        hard_update(self.critic_target, self.critic)
              
    def train(self):
        # get initial observation
        ob = self.env.reset()
        episode, episode_step = 0, 0
        epoch = 0
        
        for step in range(self.total_steps):
            # select a action
            # TODO: 1. sample from action space at the first few steps for better exploration. 2. Noise action
            action, _, _ = self.actor(self.env.get_policy_state(ob))
            
            # step the env
            _, ob2_filtered, _, done, _ = self.env.step(action)
            episode_step += 1
            
            # store experience to replay buffer: in a2cddpg, we only need ob actually
            ob_detached = {k: v.clone().detach().cpu() for k, v in ob.items()} # crucial to prevent out of gpu memory
            for i in range(list(ob_detached.values())[0].shape[0]):
                single_sample = {k: v[i:i+1,...] for k, v in ob.items()}
                self.buffer.store(single_sample)
                
            ob = ob2_filtered
            
            # end of trajectory handling
            if done or (episode_step == self.opt.max_step):
                if step > self.opt.warmup:
                    self.updaet_policy(episode, step)
                
                ob = self.env.reset()
                episode += 1
                episode_step = 0

            # end of epoch handling
            if (step + 1) % self.opt.steps_per_epoch == 0:
                epoch = (step+1) // self.opt.steps_per_epoch
                epoch += 1
                
                # if self.evaluator is not None:
                #     self.evaluator(env, agent.select_action, step, opt.loop_penalty)

                if (epoch % self.opt.save_freq == 0) or (epoch == self.opt.epochs):
                    prRed('Saving model at Step_{:07d}...'.format(step))
                    self.save_model(self.opt.save_path, step)
    
    
    def updaet_policy(self, episode, step):
        tot_Q, tot_value_loss, tot_dist_entropy = 0, 0, 0
        lr = lr_scheduler(step)
        
        for _ in range(self.opt.episode_train_times):
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
       
        policy_state = self.env.get_policy_state(state)
        eval_state = self.env.get_eval_state(state)
        
        action, action_log_prob, dist_entropy = self.actor(policy_state)
        state2, reward = self.env.forward(state, action)
        
        reward -= self.opt.loop_penalty
        eval_state2 = self.env.get_eval_state(state2)
        
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
        ddpg_loss = V_next + reward
        
        entroy_regularization = dist_entropy
        
        policy_loss = - (a2c_loss + ddpg_loss + entroy_regularization).mean()
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
        batch = {}
        for s in states:
            for k, v in s.items():
                if k not in batch.keys():
                    batch[k] = []
                batch[k].append(v)
        for k, v in batch.items():
            batch[k] = torch.cat(v, dim=0).detach().to(self.device)
        return batch
        
    def save_model(self):
        pass
    
    def load_model(self):
        pass
