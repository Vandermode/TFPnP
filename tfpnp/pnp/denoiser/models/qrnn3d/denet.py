import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DeNet(nn.Module):
    def __init__(self, in_channels=10, kernel_size=3, init_weights=True):
        super(DeNet, self).__init__()
        layers = []
        out_channels = 64
        # add CR
        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # add CBR1 to CBR16
        in_channels = out_channels
        for out_channels, dilation in cfg:
            if dilation == 1:
                padding = 1
            elif dilation == 2:
                padding = 2
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=padding,
                                    dilation=dilation, bias=False)
                          )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # add C
        layers.append(nn.Conv2d(in_channels=64, out_channels=10,
                      kernel_size=kernel_size, padding=1, bias=False)
                      )
        self.denet = nn.Sequential(*layers)
        #if init_weights:
        #   self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.denet(x)
        return y - out

    def _initialize_weights(self):
        print("===> Start initializing weights")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

"""
each tuple unit: [out_channels, dilation]
"""
cfg = ([64, 1], [64, 1], [64, 1],
       [128, 1], [128, 1], [128, 1],
       [256, 2], [256, 2], [256, 2],
       [128, 1], [128, 1], [128, 1],
       [64, 1], [64, 1], [64, 1], [64, 1])
