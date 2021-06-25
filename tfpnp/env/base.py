import torch
import torch.nn.functional as F
import numpy as np

from ..data.util import dict_to_device
from ..utils.misc import torch2img255


class Env:
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError
    
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the agent
        
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError


class DifferentiableEnv(Env):
    def forward(self, state, action):
        raise NotImplementedError
       

class PnPEnv(DifferentiableEnv):
    def __init__(self, data_loader, solver, max_step, device):
        super(PnPEnv, self).__init__() 
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader) if data_loader is not None else None
        self.device = device
        
        self.solver = solver
        
        self.max_step = max_step
        self.cur_step = 0
        
        self.state = None
        self.last_metric = 0
        self.metric_fn = torch_psnr
    
    ###################################################
    #   Abstract methods 
    ###################################################
    
    def get_policy_state(self, state):
        raise NotImplementedError
    
    def get_eval_state(self, state):
        raise NotImplementedError
    
    def _get_attribute(self, state, key):
        raise NotImplementedError
    
    def _build_next_state(self, state, solver_state):
        raise NotImplementedError
    
    def _observation(self):
        raise NotImplementedError
    
    ###################################################
    #   Basic APIs 
    ###################################################
    
    def reset(self, data=None):
        self.cur_step = 0  # # of step in an episode
        
        # load a new batch of data
        if data is None:
            try:
                data = self.data_iterator.next()
            except StopIteration:
                self.data_iterator = iter(self.data_loader)
                data = self.data_iterator.next()

        # move data to device
        data = dict_to_device(data, self.device)
        
        # get inital solver states
        solver_state = self.solver.reset(data)
        data.update({'solver': solver_state})
        
        # construct state of time step
        B,_,W,H = data['gt'].shape
        T = torch.ones([B, 1, W, H, 2], dtype=torch.float32, device=self.device) * self.cur_step / self.max_step
        data.update({'T': T})

        self.state = data
        self.idx_left = torch.arange(0, B).to(self.device)        
        self.last_metric = self._compute_metric()

        return self._observation()
    
    def step(self, action):
        self.cur_step += 1
        
        # perform one step using solver and update state
        with torch.no_grad():
            f = lambda x: x[self.idx_left, ...]
            inputs = (f(self.state['solver']), map(f, self.solver.filter_additional_input(self.state)))
            parameters = self.solver.filter_hyperparameter(action)
            solver_state = self.solver(inputs, parameters)
        
        self.state['T'] = torch.ones_like(self.state['T']) * self.cur_step / self.max_step
        self.state['output'][self.idx_left, ...] = self.solver.get_output(solver_state)
        self.state['solver'][self.idx_left, ...] = solver_state
            
        # compute reward
        reward = self._compute_reward()
        
        ob = self._observation()
        
        # update idx of items that should be processed in the next iteration
        idx_stop = action['idx_stop'] 
        self.idx_left = self.idx_left[idx_stop == 0]
        all_done = len(self.idx_left) == 0
        done = idx_stop.detach()
          
        if self.cur_step == self.max_step:
            all_done = True
            done = torch.ones_like(idx_stop)
        
        ob_masked = self._observation()
        
        return ob, ob_masked, reward, all_done, {'done': done}
    
    def forward(self, state, action):
        output = self._get_attribute(state, 'output')
        gt = self._get_attribute(state, 'gt')
        inputs = self._get_attribute(state, 'solver_input')

        parameters = self.solver.filter_hyperparameter(action)
        solver_state = self.solver(inputs, parameters)
        output2 = self.solver.get_output(solver_state)

        # compute reward
        reward = self.metric_fn(output2, gt) - self.metric_fn(output, gt)
    
        return self._build_next_state(state, solver_state), reward
    
    def get_images(self, state, pre_process=torch2img255):
        input = pre_process(self._get_attribute(state, 'input'))        
        output = pre_process(self._get_attribute(state, 'output'))
        gt = pre_process(self._get_attribute(state, 'gt'))

        return input, output, gt
    
    ###################################################
    #   Private utils
    ###################################################
    
    def _compute_metric(self):
        output = self.state['output'].clone().detach()
        gt = self.state['gt'].clone().detach()
        return self.metric_fn(output, gt)
    
    def _compute_reward(self):
        metric = self._compute_metric()
        reward = metric - self.last_metric
        self.last_metric = metric
        return reward


def torch_psnr(output, gt):
    N = output.shape[0]
    output = torch.clamp(output, 0, 1)
    mse = torch.mean(F.mse_loss(output.view(N, -1), gt.view(N, -1), reduction='none'), dim=1)
    psnr = 10 * torch.log10((1 ** 2) / mse)
    return psnr.unsqueeze(1)