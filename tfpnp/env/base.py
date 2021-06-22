import torch
from torch.nn import parameter
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from ..data.dataset import dict_to_device
from ..util.misc import to_numpy
from ..pnp.util.transforms import complex2real, ifft2

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


class DifferentiableEnv(Env):
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
        self.last_metric = 0
        self.metric_fn = torch_psnr
    
    def get_policy_state(self, state):
        raise NotImplementedError
    
    def get_eval_state(self, state):
        raise NotImplementedError
    
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
        data = dict_to_device(data, device)
        
        # get inital solver states
        solver_state = self.solver.reset(data)
        data.update({'solver': solver_state})
        
        # construct state of time step
        B,_,W,H = data['gt'].shape
        T = torch.ones([B, 1, W, H, 2], dtype=torch.float32, device=device) * self.cur_step / self.max_step
        data.update({'T': T})

        self.state = data
        self.idx_left = torch.arange(0, B).to(device)        
        self.last_metric = self.compute_metric()

        return self._observation()
    
    def compute_metric(self):
        output = self.state['output'].clone().detach()
        gt = self.state['gt'].clone().detach()
        return self.metric_fn(output, gt)
    
    def compute_reward(self):
        metric = self.compute_metric()
        reward = metric - self.last_metric
        self.last_metric = metric
        return reward
    
    def filter(self, data):
        return data[self.idx_left, ...]
    
    def step(self, action):
        self.cur_step += 1
        
        # perform one step using solver and update state
        with torch.no_grad():
            f = self.filter
            inputs = (f(self.state['solver']), map(f, self.solver.filter_additional_input(self.state)))
            parameters = self.solver.filter_hyperparameter(action)
            solver_state = self.solver(inputs, parameters)
            self.state['T'] = torch.ones_like(self.state['T']) * self.cur_step / self.max_step
            self.state['output'][self.idx_left, ...] = self.solver.get_output(solver_state)
            self.state['solver'][self.idx_left, ...] = solver_state
            
        # compute reward
        reward = self.compute_reward()
        
        ob = self._observation()
        
        # update idx of items that should be left to be process
        idx_stop = action['idx_stop'] 
        self.idx_left = self.idx_left[idx_stop == 0]
        all_done = len(self.idx_left) == 0
        
        ob_masked = self._observation()
          
        if self.cur_step == self.max_step:
            all_done = True
            done = torch.ones_like(idx_stop)
        else:
            done = idx_stop.detach()
        
        return ob, ob_masked, reward, all_done, {'done': done}
    
    
    def forward(self, inputs, output, gt, action):
        last_psnr = torch_psnr(output, gt)

        parameters = self.solver.filter_hyperparameter(action)
        solver_state = self.solver(inputs, parameters)
        output2 = self.solver.get_output(solver_state)

        # compute reward
        psnr = torch_psnr(output2, gt)
        reward = (psnr - last_psnr)
    
        return solver_state, reward
    
    def get_images(self, state):
        def _pre_img(img):
            img = to_numpy(img[0,...])
            img = np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8), 3, axis=0)
            return img

        input = _pre_img(complex2real(state[:,5:6,...]))        
        output = _pre_img(complex2real(state[:,1:2,...]))
        gt = _pre_img(complex2real(state[:,0:1,...]))

        return input, output, gt
    
    # def _observation(self, state=None):
    #     if state is None:
    #         s = {k: v[self.idx_left, ...] for k,v in deepcopy(self.state).items()}
    #         return s
    #     else:
    #         return state


def torch_psnr(output, gt):
    N = output.shape[0]
    output = torch.clamp(output, 0, 1)
    mse = torch.mean(F.mse_loss(output.view(N, -1), gt.view(N, -1), reduction='none'), dim=1)
    psnr = 10 * torch.log10((1 ** 2) / mse)
    return psnr.unsqueeze(1)