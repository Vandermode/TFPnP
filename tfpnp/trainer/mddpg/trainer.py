import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tensorboardX.writer import SummaryWriter
import time
import os

from ...data.batch import Batch
from ...eval import Evaluator
from ...env import PnPEnv
from ...utils.misc import DataParallel, soft_update, hard_update
from ...utils.rpm import ReplayMemory
from ...utils.log import Logger, COLOR
from ...policy.sync_batchnorm import DataParallelWithCallback
from .critic import ResNet_wobn

DataParallel = DataParallelWithCallback


class MDDPGTrainer:
    def __init__(self, 
                 opt, 
                 env: PnPEnv, 
                 policy, 
                 lr_scheduler,
                 device, 
                 log_dir, 
                 evaluator: Evaluator = None,
                 enable_tensorboard=False, 
                 logger=None
                 ):

        self.data_parallel = True if torch.cuda.device_count() > 1 else False
        self.opt = opt
        self.env = env
        self.actor = policy
        self.critic = ResNet_wobn(policy.in_dim, 18, 1).to(device)
        self.critic_target = ResNet_wobn(policy.in_dim, 18, 1).to(device)
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.writer = SummaryWriter(os.path.join(log_dir, 'tfborad')) if enable_tensorboard else None
        self.device = device
        self.logger = Logger(log_dir) if logger is None else logger

        # TODO: estimate actual needed memory to prevent OOM error.
        self.buffer = ReplayMemory(opt.rmsize * opt.max_episode_step)

        self.optimizer_actor = Adam(self.actor.parameters())
        self.optimizer_critic = Adam(self.critic.parameters())

        self.criterion = nn.MSELoss()   # criterion for value loss

        hard_update(self.critic_target, self.critic)

        self.choose_device()

        self.start_step = 1

    def train(self):
        # get initial observation
        ob = self.env.reset()

        hidden = hidden_full = self.actor.init_state(ob.shape[0]).to(self.device)  # TODO: add RNN support
        episode, episode_step = 0, 0
        time_stamp = time.time()

        best_eval_psnr = 0
        
        for step in range(self.start_step, self.opt.train_steps+1):
            # select a action
            action, hidden = self.run_policy(self.env.get_policy_ob(ob), hidden)

            # step the env
            _, ob2_masked, _, done, _ = self.env.step(action)
            episode_step += 1

            # store experience to replay buffer: in a2cddpg, we only need ob actually
            self.save_experience(ob, hidden)

            ob = ob2_masked
            hidden = hidden_full[self.env.idx_left, ...]

            # end of trajectory handling
            if done or (episode_step == self.opt.max_episode_step):

                if step > self.opt.warmup:
                    if self.evaluator is not None and (episode+1) % self.opt.validate_interval == 0:
                        eval_psnr = self.evaluator.eval(self.actor, step)
                        if eval_psnr > best_eval_psnr:
                            best_eval_psnr = eval_psnr
                            self.save_model(self.opt.output, 'best_{:.2f}'.format(best_eval_psnr))
                            self.save_model(self.opt.output, 'best')
                        self.save_model(self.opt.output)
                        

                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                result = {'Q': 0, 'dist_entropy': 0, 'critic_loss': 0}

                if step > self.opt.warmup:
                    result, tb_result = self._update_policy(self.opt.episode_train_times,
                                                            self.opt.env_batch,
                                                            step=step)
                    if self.writer is not None:
                        for k, v in tb_result.items():
                            self.writer.add_scalar(f'train/{k}', v, step)

                # handle logging of training results
                fmt_str = '#{}: Steps: {} - RPM[{}/{}] | interval_time: {:.2f} | train_time: {:.2f} | {}'
                fmt_result = ' | '.join([f'{k}: {v:.2f}' for k, v in result.items()])
                self.logger.log(fmt_str.format(episode, step, self.buffer.size(), self.buffer.capacity,
                                               train_time_interval, time.time()-time_stamp, fmt_result))

                # reset state for next episode
                ob = self.env.reset()
                hidden = hidden_full = self.actor.init_state(ob.shape[0]).to(self.device)
                episode += 1
                episode_step = 0
                time_stamp = time.time()

            # save model
            if step % self.opt.save_freq == 0 or step == self.opt.train_steps:
                self.evaluator.eval(self.actor, step)
                self.logger.log('Saving model at Step_{:07d}...'.format(step), color=COLOR.RED)
                self.save_model(self.opt.output, step)

    def _update_policy(self, episode_train_times, env_batch, step):
        self.actor.train()

        tot_Q, tot_value_loss, tot_dist_entropy = 0, 0, 0
        tot_actor_norm, tot_critic_norm = 0, 0
        lr = self.lr_scheduler(step)

        for _ in range(episode_train_times):
            samples = self.buffer.sample_batch(env_batch)
            Q, value_loss, dist_entropy, actor_norm, critic_norm = self._update(samples=samples, lr=lr)

            tot_Q += Q
            tot_value_loss += value_loss
            tot_dist_entropy += dist_entropy
            tot_actor_norm += actor_norm
            tot_critic_norm += critic_norm

        mean_Q = tot_Q / episode_train_times
        mean_dist_entropy = tot_dist_entropy / episode_train_times
        mean_value_loss = tot_value_loss / episode_train_times
        mean_actor_norm = tot_actor_norm / episode_train_times
        mean_critic_norm = tot_critic_norm / episode_train_times

        tensorboard_result = {'critic_lr': lr['critic'], 'actor_lr': lr['actor'],
                              'Q': mean_Q, 'dist_entropy': mean_dist_entropy, 'critic_loss': mean_value_loss,
                              'actor_norm': mean_actor_norm, 'critic_norm': mean_critic_norm}
        result = {'Q': mean_Q, 'dist_entropy': mean_dist_entropy, 'closs': mean_value_loss,
                   'anorm': mean_actor_norm, 'cnorm': mean_critic_norm}

        return result, tensorboard_result

    def _update(self, samples, lr: dict):
        # update learning rate
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = lr['actor']
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = lr['critic']

        # convert list of named tuple into named tuple of batch
        ob = self.convert2batch(samples)
        hidden = ob.hidden

        policy_ob = self.env.get_policy_ob(ob)

        action, action_log_prob, dist_entropy, _ = self.actor(policy_ob, None, True, hidden)

        ob2, reward = self.env.forward(ob, action)
        reward -= self.opt.loop_penalty

        eval_ob = self.env.get_eval_ob(ob)
        eval_ob2 = self.env.get_eval_ob(ob2)

        # compute actor critic loss for discrete action
        V_cur = self.critic(eval_ob)
        with torch.no_grad():
            V_next_target = self.critic_target(eval_ob2)
            V_next_target = (
                self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next_target
            Q_target = V_next_target + reward
        advantage = (Q_target - V_cur).clone().detach()
        a2c_loss = action_log_prob * advantage

        # compute ddpg loss for continuous actions
        V_next = self.critic(eval_ob2)
        V_next = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next
        ddpg_loss = V_next + reward

        # compute entroy regularization
        entroy_regularization = dist_entropy

        policy_loss = - (a2c_loss + ddpg_loss + self.opt.lambda_e * entroy_regularization).mean()
        value_loss = self.criterion(Q_target, V_cur)

        # perform one step gradient descent
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 50)
        self.optimizer_actor.step()

        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
        self.optimizer_critic.step()

        # soft update target network
        soft_update(self.critic_target, self.critic, self.opt.tau)

        return -policy_loss.item(), value_loss.item(), entroy_regularization.mean().item(), actor_norm.item(), critic_norm.item()

    def run_policy(self, ob, hidden=None):
        self.actor.eval()
        with torch.no_grad():
            action, _, _, hidden = self.actor(ob, None, True, hidden)
            # action, _, _, hidden = self.actor(state=ob, idx_stop=None, train=True, hidden=hidden)
        self.actor.train()
        return action, hidden

    def save_experience(self, ob, hidden):
        for k, v in ob.items():
            if isinstance(v, torch.Tensor):
                ob[k] = ob[k].clone().detach().cpu()

        hidden = hidden.clone().detach().cpu()
        ob['hidden'] = hidden

        B = ob.shape[0]
        for i in range(B):
            self.buffer.store(ob[i])

    def convert2batch(self, obs):
        batch = Batch.stack(obs)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    def save_model(self, path, step=None):
        path = os.path.join(path, 'ckpt')
        os.makedirs(path, exist_ok=True)
        if self.data_parallel:
            self.actor = self.actor.module
            self.critic = self.critic.module
            self.critic_target = self.critic_target.module

        self.actor.cpu()
        self.critic.cpu()
        if step is None:
            torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(path))
            torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(path))
        else:
            postfix = step if isinstance(step, str) else '{:07d}'.format(step)
            torch.save(self.actor.state_dict(),
                       '{}/actor_{}.pkl'.format(path, postfix))
            torch.save(self.critic.state_dict(),
                       '{}/critic_{}.pkl'.format(path, postfix))

        self.choose_device()

    def load_model(self, path, step=None):
        if step is None:
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        else:
            self.start_step = step
            self.actor.load_state_dict(torch.load('{}/actor_{:07d}.pkl'.format(path, step)))
            self.critic.load_state_dict(torch.load('{}/critic_{:07d}.pkl'.format(path, step)))

    def choose_device(self):
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        if self.data_parallel and type(self.actor) is not DataParallel:
            self.actor = DataParallel(self.actor)
            self.critic = DataParallel(self.critic)
            self.critic_target = DataParallel(self.critic_target)
