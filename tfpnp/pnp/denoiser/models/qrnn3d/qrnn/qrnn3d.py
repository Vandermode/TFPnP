import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np

from functools import partial

if __name__ == '__main__':
    from combinations import *
    from utils import *
else:
    from .combinations import *
    from .utils import *

"""F pooling"""


class QRNN3DLayer(nn.Module):
    """
    Inputs: B,C,B,W,H
    """

    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        # conv is convolution unit: output channel = input channel * 2 (single direction) or * 3 (bi-directional)
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        return torch.cat(h_time, dim=2)

    def extra_repr(self):
        return 'act={}'.format(self.act)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'none':
            return Z, F1.sigmoid(), F2.sigmoid()
        else:
            raise NotImplementedError

    def forward(self, inputs, fname=None):
        h = None
        Z, F1, F2 = self._conv_step(inputs)
        hsl = []
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep            
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None
        for time, (z, f) in enumerate((zip(
                reversed(zs), reversed(F2.split(1, 2))
        ))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {'z': Z, 'fl': F1, 'fr': F2, 'hsl': hsl, 'hsr': hsr}
            torch.save(stats_dict, fname)
        return hsl + hsr


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 3, k, s, p, bn=bn), act=act)


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act='tanh'):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels * 3, k, s, p, bias=bias, bn=bn),
            act=act)


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True, act='tanh'):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels,
            BasicUpsampleConv3d(in_channels, hidden_channels * 2, k, s, p, upsample, bn=bn), act=act)


QRNN3DEncoder = partial(
    QRNN3DEncoder,
    QRNNConv3D=QRNNConv3D)

QRNN3DDecoder = partial(
    QRNN3DDecoder,
    QRNNDeConv3D=QRNNDeConv3D,
    QRNNUpsampleConv3d=QRNNUpsampleConv3d)

QRNNREDC3D = partial(
    QRNNREDC3D,
    BiQRNNConv3D=BiQRNNConv3D,
    BiQRNNDeConv3D=BiQRNNDeConv3D,
    QRNN3DEncoder=QRNN3DEncoder,
    QRNN3DDecoder=QRNN3DDecoder
)
