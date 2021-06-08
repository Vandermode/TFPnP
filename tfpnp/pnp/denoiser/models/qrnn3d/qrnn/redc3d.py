import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if __name__ == '__main__':
    from combinations import *
else:
    from .combinations import *


class REDC3D(torch.nn.Module):
    """Residual Encoder-Decoder Convolution 3D
    Args:
        downsample: downsample times, None denotes no downsample"""
    def __init__(self, in_channels, channels, num_half_layer, downsample=None):
        super(REDC3D, self).__init__()
        # Encoder
        assert downsample is None or 0 < downsample <= num_half_layer
        interval = num_half_layer // downsample if downsample else num_half_layer+1

        self.feature_extractor = BNReLUConv3d(in_channels, channels)        
        self.encoder = nn.ModuleList()
        for i in range(1, num_half_layer+1):
            if i % interval:
                encoder_layer = BNReLUConv3d(channels, channels)
            else:
                encoder_layer = BNReLUConv3d(channels, 2*channels, k=3, s=(1,2,2), p=1)
                channels *= 2
            self.encoder.append(encoder_layer)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(1,num_half_layer+1):
            if i % interval:                
                decoder_layer = BNReLUDeConv3d(channels, channels)
            else:
                decoder_layer = BNReLUUpsampleConv3d(channels, channels//2)
                channels //= 2
            self.decoder.append(decoder_layer)
        self.reconstructor = BNReLUDeConv3d(channels, in_channels)
    
    def forward(self, x):
        num_half_layer = len(self.encoder)
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        for i in range(num_half_layer-1):
            out = self.encoder[i](out)
            xs.append(out)
        out = self.encoder[-1](out)
        out = self.decoder[0](out)        
        for i in range(1, num_half_layer):
            out = out + xs.pop()
            out = self.decoder[i](out)
        out = out + xs.pop()
        out = self.reconstructor(out)
        out = out + xs.pop()
        return out
