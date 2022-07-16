import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        