import random
import numpy as np
import torch
import argparse
import os

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    # parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Tuning-free Plug-and-Play Proximal Algorithm')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp', default='baseline', type=str, help='name of experiment')
        self.parser.add_argument('--warmup', default=10, type=int, help='timestep without training but only filling the replay memory')
        self.parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
        self.parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
        self.parser.add_argument('--env_batch', default=48, type=int, help='concurrent environment number')
        self.parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--max_step', default=6, type=int, help='max length for episode')
        self.parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise') # 0.04
        self.parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
        self.parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
        
        self.parser.add_argument('--epochs', default=10, type=int, help='number of epochs for training')
        self.parser.add_argument('--steps_per_epoch', default=100, type=int, help='number of steps per epoch')
        self.parser.add_argument('--save_freq', default=1000, type=int, help='number of steps per epoch')
        self.parser.add_argument('--eval_per_episode', default=10, type=int, help='number of steps per epoch')
        
        self.parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
        self.parser.add_argument('--resume', '-r', default=None, type=str, help='Resuming model path')
        self.parser.add_argument('--resume_step', '-rs', default=None, type=int, help='Resuming model step')
        self.parser.add_argument('--output', default='./checkpoints', type=str, help='resuming model path for testing')
        self.parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
        self.parser.add_argument('--seed', default=1234, type=int, help='random seed')
        self.parser.add_argument('--loop_penalty', '-lp', type=float, default=0.05, help='penalty of loop')        
        self.parser.add_argument('--action_pack', '-ap', type=int, default=5, help='pack of action')
        self.parser.add_argument('--lambda_e', '-le', type=float, default=0.05, help='penalty of loop')
        self.parser.add_argument('--denoiser', type=str, default='unet', help='denoising network')
        self.parser.add_argument('--solver', type=str, default='admm', help='invoked solver')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt.output = get_output_folder(opt.output, opt.exp)
        print('[i] Exp dir: {}'.format(opt.output))

        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)

        if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
        if torch.cuda.device_count() > 1:
            print("[i] Use", torch.cuda.device_count(), "GPUs...")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        self.opt = opt

        return opt


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Tuning-free Plug-and-Play Proximal Algorithm')
        self.initialized = False

    def initialize(self):
        # hyper-parameter
        self.parser.add_argument('--exp', default='baseline', type=str, help='name of experiment')
        self.parser.add_argument('--warmup', default=100, type=int, help='timestep without training but only filling the replay memory')
        self.parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
        self.parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
        self.parser.add_argument('--env_batch', default=48, type=int, help='concurrent environment number')
        self.parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--max_step', default=6, type=int, help='max length for episode')
        self.parser.add_argument('--resume', '-r', default=None, type=str, help='Resuming model path')
        self.parser.add_argument('--resume_step', '-rs', default=None, type=int, help='Resuming model step')
        self.parser.add_argument('--output', default='./checkpoints', type=str, help='resuming model path for testing')
        self.parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
        self.parser.add_argument('--seed', default=1234, type=int, help='random seed')
        self.parser.add_argument('--loop_penalty', '-lp', type=float, default=0.05, help='penalty of loop')        
        self.parser.add_argument('--action_pack', '-ap', type=int, default=5, help='pack of action')
        self.parser.add_argument('--lambda_e', '-le', type=float, default=0, help='penalty of loop')        
        self.parser.add_argument('--denoiser', type=str, default='unet', help='denoising network')        
        self.parser.add_argument('--solver', type=str, default='admm', help='invoked solver')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()

        if opt.resume is None and opt.resume_step is not None:
            opt.resume = opt.exp

        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)

        if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
        if torch.cuda.device_count() > 1:
            print("[i] Use", torch.cuda.device_count(), "GPUs...")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        self.opt = opt

        return opt
