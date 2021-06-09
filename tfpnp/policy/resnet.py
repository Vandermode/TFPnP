import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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


class ResNetActor(nn.Module):
    # ADMM and HQS
    def __init__(self, num_inputs, action_bundle):
        super(ResNetActor, self).__init__() 
        self.conv = nn.Conv2d(num_inputs, out_channels=10, kernel_size=1)               
        self.actor_encoder = ResNetEncoder(10, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])
    
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*2),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop=None, stochastic=True):
        action = {}
        x = self.conv(state)
        x = self.actor_encoder(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if stochastic:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)

        half = action_deterministic.shape[1] // 2
        action['sigma_d'] = action_deterministic[:, :half] * 70 / 255
        action['mu'] = action_deterministic[:, half:] # [0-1]
        action['idx_stop'] = idx_stop
        
        return action, action_categorical_logprob, dist_entropy


class ResNetActor_PG(nn.Module):
    # PG and APG
    def __init__(self, num_inputs, action_bundle):
        super(ResNetActor_PG, self).__init__()
        self.actor_encoder = ResNetEncoder(num_inputs, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])
    
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*2),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop, stochastic):
        action = {}
        x = self.actor_encoder(state)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if stochastic:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)

        half = action_deterministic.shape[1] // 2
        action['sigma_d'] = action_deterministic[:, :half] * 70 / 255
        action['tau'] = action_deterministic[:, half:] * 2  # [0-2]

        return action, action_categorical_logprob, idx_stop, dist_entropy


class ResNetActor_APG(nn.Module):
    # APG
    def __init__(self, num_inputs, action_bundle):
        super(ResNetActor_APG, self).__init__()
        self.actor_encoder = ResNetEncoder(num_inputs, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])
    
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*3),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop, stochastic):
        action = {}
        x = self.actor_encoder(state)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if stochastic:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)

        i = action_deterministic.shape[1] // 3
        action['sigma_d'] = action_deterministic[:, :i] * 70 / 255
        action['tau'] = action_deterministic[:, i:i*2] * 2  # [0-2]
        action['beta'] = action_deterministic[:, i*2:] * 2  # [0-2]

        return action, action_categorical_logprob, idx_stop, dist_entropy


class ResNetActor_RED(nn.Module):
    # RED-ADMM
    def __init__(self, num_inputs, action_bundle):
        super(ResNetActor_RED, self).__init__()
        self.actor_encoder = ResNetEncoder(num_inputs, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])
    
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*3),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop, stochastic):
        action = {}
        x = self.actor_encoder(state)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if stochastic:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)

        i = action_deterministic.shape[1] // 3
        action['sigma_d'] = action_deterministic[:, :i] * 70 / 255
        action['mu'] = action_deterministic[:, i:2*i]               # [0-1]
        action['lamda'] = action_deterministic[:, 2*i:] * 2         # [0-2]

        return action, action_categorical_logprob, idx_stop, dist_entropy


class ResNetActor_ADMM(nn.Module):
    # Inexact PnP-ADMM
    def __init__(self, num_inputs, action_bundle):
        super(ResNetActor_ADMM, self).__init__()
        self.actor_encoder = ResNetEncoder(num_inputs, 18)

        self.fc_softmax = nn.Sequential(*[
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        ])
    
        self.fc_deterministic = nn.Sequential(*[
            nn.Linear(512, action_bundle*3),
            nn.Sigmoid()
        ])

    def forward(self, state, idx_stop, stochastic):
        action = {}
        x = self.actor_encoder(state)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        action_probs = self.fc_softmax(x)        
        action_deterministic = self.fc_deterministic(x)
        
        # discrete action
        dist_categorical = Categorical(action_probs)
        dist_entropy = dist_categorical.entropy().unsqueeze(1)

        if idx_stop is None:
            if stochastic:
                # idx_stop = torch.argmax(action_probs, dim=1)
                idx_stop = dist_categorical.sample()
            else:
                idx_stop = torch.argmax(action_probs, dim=1)
        
        action_categorical_logprob = dist_categorical.log_prob(idx_stop).unsqueeze(1)

        i = action_deterministic.shape[1] // 3
        action['sigma_d'] = action_deterministic[:, :i] * 70 / 255
        action['mu'] = action_deterministic[:, i:2*i]             # [0-1]
        action['tau'] = action_deterministic[:, 2*i:] * 2         # [0-2]

        return action, action_categorical_logprob, idx_stop, dist_entropy
