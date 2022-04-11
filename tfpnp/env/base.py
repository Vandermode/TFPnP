import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from ..utils.misc import torch2img255, apply_recursive
from ..pnp import PnPSolver
from ..data.batch import Batch


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
    def __init__(self, data_loader: DataLoader, solver: PnPSolver, max_episode_step, data_transform=None):
        super(PnPEnv, self).__init__()
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader) if data_loader is not None else None
        self.device = torch.device('cpu')
        self.data_transform = data_transform

        self.solver = solver

        self.max_episode_step = max_episode_step
        self.cur_step = 0

        self.state = None
        self.last_metric = 0
        self.metric_fn = torch_psnr

    ################################################################################
    #   Abstract methods
    #
    #   You need to implement all the following methods in your environment to
    #   make it compatible with MDDPGTrainer.
    #################################################################################

    def get_policy_ob(self, ob: Batch):
        """ Extract the input for policy network from the observation.

            Args:
                ob (Batch): the observation.

            Returns:
                A torch tensor for policy network.
        """
        raise NotImplementedError

    def get_eval_ob(self, ob: Batch):
        """ Extract the input for critic network from the observation.

            Args:
                ob (Batch): the observation.

            Returns:
                A torch tensor for evalutaion network, such as critic in A2C.
        """
        raise NotImplementedError

    def _get_attribute(self, ob: Batch, key: str):

        raise NotImplementedError

    def _build_next_ob(self, ob: Batch, solver_state):
        """ Build next observation from current observatio and solver state.

        Args:
            ob (Batch): the current observation.
            solver_state (Any): the current solver state.

        Returns:
            A Batch contains the next observation.
        """
        raise NotImplementedError

    def _observation(self):
        """ Construct the observation from the internal state of the env, which can
            be accessed by `self.state`.

            Returns:
                A Batch contains the observation. Make sure each element's first dim is Batch size.
        """
        raise NotImplementedError

    #################################################################################
    #   Basic APIs
    #
    #   Do not modify the following methods unless you understand what you are doing.
    #################################################################################

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
        if self.data_transform is not None:
            data = self.data_transform(data)

        def to_device(x):
            if isinstance(x, torch.Tensor): return x.to(self.device)
        data = apply_recursive(to_device, data)

        # get inital solver states
        solver_state = self.solver.reset(data)
        data.update({'solver': solver_state})

        # construct state of time step
        B, _, W, H = data['gt'].shape
        # sigma_n = torch.ones_like(data['gt']) * data['sigma_n'].view(B, 1, 1, 1)
        T = torch.ones([B, 1, W, H], dtype=torch.float32,
                       device=self.device) * self.cur_step / self.max_episode_step
        data.update({'T': T})

        self.state = data
        self.idx_left = torch.arange(0, B).to(self.device)
        self.last_metric = self._compute_metric()

        return self._observation()

    def step(self, action):
        self.cur_step += 1

        # perform one step using solver and update state
        with torch.no_grad():
            def f(x): return x[self.idx_left, ...]
            inputs = (f(self.state['solver']),
                      #   f(self.solver.filter_aux_inputs(self.state))
                      apply_recursive(f, self.solver.filter_aux_inputs(self.state))
                      )
            parameters = self.solver.filter_hyperparameter(action)
            solver_state = self.solver(inputs, parameters)

        self.state['T'] = torch.ones_like(self.state['T']) * self.cur_step / self.max_episode_step
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

        if self.cur_step == self.max_episode_step:
            all_done = True
            done = torch.ones_like(idx_stop)

        ob_masked = self._observation()

        return ob, ob_masked, reward, all_done, {'done': done}

    def forward(self, ob, action):
        output = self._get_attribute(ob, 'output')
        gt = self._get_attribute(ob, 'gt')
        inputs = self._get_attribute(ob, 'solver_input')

        parameters = self.solver.filter_hyperparameter(action)
        solver_state = self.solver(inputs, parameters)
        output2 = self.solver.get_output(solver_state)

        # compute reward
        # print(gt.shape, output.shape, output2.shape)
        reward = self.metric_fn(output2, gt) - self.metric_fn(output, gt)

        return self._build_next_ob(ob, solver_state), reward

    def get_images(self, ob, pre_process=torch2img255):
        input = pre_process(self._get_attribute(ob, 'input'))
        output = pre_process(self._get_attribute(ob, 'output'))
        gt = pre_process(self._get_attribute(ob, 'gt'))

        return input, output, gt

    def to(self, device):
        if not isinstance(device, torch.device):
            raise TypeError('device must be torch.device, but got {}'.format(type(device)))
        self.device = device
        return self

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
