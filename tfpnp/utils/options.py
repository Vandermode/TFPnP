import random
import numpy as np
import torch
import argparse
from .misc import get_output_folder


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Tuning-free Plug-and-Play Proximal Algorithm')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp', default='csmri_admm_5x6_48', type=str, help='name of experiment')
        self.parser.add_argument('--warmup', default=20, type=int, help='timestep without training but only filling the replay memory')
        self.parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
        self.parser.add_argument('--rmsize', default=480, type=int, help='replay memory size')
        self.parser.add_argument('--env_batch', default=48, type=int, help='concurrent environment number')
        self.parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--max_episode_step', default=6, type=int, help='max length for episode')                
        self.parser.add_argument('--train_steps', default=15000, type=int, help='number of train iters')
        self.parser.add_argument('--validate_interval', default=1, type=int, help='how many episodes to perform a validation')
        self.parser.add_argument('--save_freq', default=1000, type=int, help='number of steps per epoch')        
        self.parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
        self.parser.add_argument('--resume', '-r', default=None, type=str, help='Resuming model path')
        self.parser.add_argument('--resume_step', '-rs', default=None, type=int, help='Resuming model step')
        self.parser.add_argument('--eval', action='store_true', help='eval mode')
        self.parser.add_argument('--seed', default=1234, type=int, help='random seed')
        self.parser.add_argument('--num_workers', default=8, type=int, help='number of workers on dataloader')        
        self.parser.add_argument('--loop_penalty', '-lp', type=float, default=0.05, help='penalty of loop')        
        self.parser.add_argument('--action_pack', '-ap', type=int, default=5, help='pack of action')
        self.parser.add_argument('--lambda_e', '-le', type=float, default=0.05, help='penalty of loop')
        self.parser.add_argument('--denoiser', type=str, default='unet', help='denoising network')
        self.parser.add_argument('--solver', type=str, default='admm', help='invoked solver')
        self.parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt.output = get_output_folder('log', opt.exp)
        print('[i] Exp dir: {}'.format(opt.output))

        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)

        if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
        if torch.cuda.device_count() > 1:
            print("[i] Use", torch.cuda.device_count(), "GPUs...")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        opt.num_workers = 0 if opt.debug else opt.num_workers        
        self.opt = opt

        return opt
