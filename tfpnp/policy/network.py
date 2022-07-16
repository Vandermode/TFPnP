from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import PolicyNetwork
from .sync_batchnorm import SynchronizedBatchNorm2d

# norm = nn.BatchNorm2d
# norm = nn.InstanceNorm2d
norm = SynchronizedBatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    return (nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = (nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.conv2 = (nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv3 = (nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False))
        self.bn1 = norm(planes)
        self.bn2 = norm(planes)
        self.bn3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, num_inputs, depth):
        super(ResNetEncoder, self).__init__()
        block, num_blocks = cfg(depth)
        self.in_planes = 64

        self.conv1 = conv3x3(num_inputs, 64, 2) 
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x8        
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x4       

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)        

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetActorBase(PolicyNetwork):
    def __init__(self, num_inputs, action_bundle, num_actions):
        super().__init__(num_inputs)
        self.num_actions = num_actions
        self.action_range = None
        self.action_bundle = action_bundle
        
        self.actor_encoder = ResNetEncoder(num_inputs, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])

        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*num_actions),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop, train, hidden):
        x = self.actor_encoder(state)
        # x = F.avg_pool2d(x, 4)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if train:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)
        action = self.action_mapping(action_deterministic)
        action['idx_stop'] = idx_stop

        return action, action_categorical_logprob, dist_entropy, hidden

    def action_mapping(self, action_deterministic):
        num_actions = self.num_actions
        action_range = self.action_range
        
        chunk_size = int(action_deterministic.shape[1] // num_actions)
        action_values = torch.split(action_deterministic, chunk_size, dim=1)
        action = OrderedDict()
        for i, key in enumerate(action_range):
            action[key] = action_values[i] * action_range[key]['scale'] \
                + action_range[key]['shift']

        return action

    #TODO add RNN support
    def init_state(self, B):
        return torch.zeros(B)
    

class ResNetActor_ADMM(ResNetActorBase):
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+3, action_bundle, 2)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'mu': {'scale': 1, 'shift': 0}                
            })
        self.action_range = action_range


class ResNetActor_HQS(ResNetActorBase):
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+2, action_bundle, 2)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'mu': {'scale': 1, 'shift': 0}
            })
        self.action_range = action_range


class ResNetActor_PG(ResNetActorBase):
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+1, action_bundle, 2)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'tau': {'scale': 2, 'shift': 0}
            })
        self.action_range = action_range


class ResNetActor_APG(ResNetActorBase):
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+2, action_bundle, 3)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'tau': {'scale': 2, 'shift': 0},
                'beta': {'scale': 2, 'shift': 0},
            })            
        self.action_range = action_range


class ResNetActor_RED(ResNetActorBase):
    # RED-ADMM
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+3, action_bundle, 3)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'mu': {'scale': 1, 'shift': 0},
                'lamda': {'scale': 2, 'shift': 0},
            })            
        self.action_range = action_range


class ResNetActor_IADMM(ResNetActorBase):
    # Inexact PnP-ADMM
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+3, action_bundle, 3)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 70 / 255, 'shift': 0},
                'mu': {'scale': 1, 'shift': 0},
                'tau': {'scale': 2, 'shift': 0},
            })
        self.action_range = action_range


class ResNetActor_AMP(ResNetActorBase):
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+2, action_bundle, 1)
        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 2, 'shift': 0}, 
            })
        self.action_range = action_range


class ResNetActor_SPI(ResNetActorBase):
    # for single photon imaging
    def __init__(self, num_aux_inputs, action_bundle, action_range: Optional[OrderedDict] = None):
        super().__init__(num_aux_inputs+3, action_bundle, 2)
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, action_bundle*2),
            nn.Sigmoid()
        ])

        if action_range is None:
            action_range = OrderedDict({
                'sigma_d': {'scale': 55 / 255, 'shift': 15 / 255}, 
                'mu': {'scale': 70, 'shift': 50}
            })
        self.action_range = action_range
