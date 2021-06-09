import torch
import torch.nn.functional as F
import torch.nn as nn

from ..data.dataset import dict_to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class DifferentiableEnv(Env, nn.Module):
    def forward(self, state, action):
        raise NotImplementedError
       

class PnPEnv(DifferentiableEnv):
    def __init__(self, data_loader, solver, max_step):
        super(PnPEnv, self).__init__() 
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader)
        
        self.solver = solver
        
        self.max_step = max_step
        self.cur_step = 0
        
        self.state = None
        self.last_psnr = 0
        self.init_psnr = 0
    
    def get_policy_state(self, state):
        raise NotImplementedError
    
    def get_eval_state(self, state):
        raise NotImplementedError
    
    def reset(self):
        self.cur_step = 0  # # of step in an episode
        
        # load a new batch of data
        try:
            data = self.data_iterator.next()
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            data = self.data_iterator.next()

        # move data to device
        data = dict_to_device(data, device)
        
        # get inital solver states
        solver_state = self.solver.reset(data)
        
        # construct state of time step
        B,_,W,H = data['gt'].shape
        T = torch.ones([B, 1, W, H], dtype=torch.float32, device=device) 
        T *= self.cur_step / self.max_step
        
        # compute inital pnsr
        self.last_psnr = self.init_psnr = self._cal_psnr(data['output'], data['gt'])
        
        # concate all states 
        data.update({'T': T})
        data.update({'solver': solver_state})
        self.state = data
        
        self.idx_left = torch.arange(0, B)
        
        return self._observation()
    
    def step(self, action):
        self.cur_step += 1
        
        # perform one step using solver and update state
        inputs = (self.state['sovler'], self.solver.get_additional_input(self.state))
        solver_state = self.solver(inputs, action)
        self.state['T'] = torch.ones_like(self.state['T']) * self.cur_step / self.max_step
        self.state['output'] = self.solver.get_output(solver_state)
        self.state['solver'] = solver_state
        
        # compute reward
        psnr = self._cal_psnr(self.state['output'], self.state['gt'])
        reward = (psnr - self.last_psnr)
        
        # update idx of items that should be left to be process
        idx_stop = action['idx_stop'] 
        self.idx_left = self.idx_left[idx_stop == 0]
        all_done = len(self.idx_left) == 0
        if self.cur_step == self.max_step:
            all_done = True
            done = torch.ones_like(idx_stop)
        else:
            done = idx_stop.detach()

        ob = self._observation()
        self.state = {k: v[self.idx_left, ...] for k, v in self.state.items()}
        ob_masked = self._observation()
        
        self.last_psnr = psnr[self.idx_left, ...]
        
        return ob, ob_masked, reward, all_done, {'done': done}
    
    
    def forward(self, state, action):
        last_psnr = self._cal_psnr(state['output'], state['gt'])
          
        inputs = state['solver']
        solver_state = self.solver(inputs, action)
        state['output'] = self.solver.get_output(solver_state)
        state['solver'] = solver_state
        
        # compute reward
        psnr = self._cal_psnr(state['output'], state['gt'])
        reward = (psnr - last_psnr)
        
        ob = self._observation(state)
        return ob, reward
    
    
    def _observation(self):
        return self.state
    
    
    def _cal_psnr(self, output, gt):
        N = output.shape[0]
        output = torch.clamp(output, 0, 1)
        mse = torch.mean(F.mse_loss(output.view(N, -1), gt.view(N, -1), reduction='none'), dim=1)
        psnr = 10 * torch.log10((1 ** 2) / mse)
        return psnr.unsqueeze(1)