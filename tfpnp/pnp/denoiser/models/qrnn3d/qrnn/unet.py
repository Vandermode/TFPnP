import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from .qrnn3d import QRNNConv3D, QRNNUpsampleConv3d, BiQRNNConv3D, BiQRNNDeConv3D, QRNNDeConv3D

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(conv_block, self).__init__()
        
        self.conv1 = QRNNConv3D(in_ch, out_ch, bn=bn)
        self.conv2 = QRNNConv3D(out_ch,out_ch, bn=bn)
        self.conv_residual = QRNNConv3D(in_ch, out_ch, k=1, s=1, p=0, bn=bn)

    def forward(self, x, reverse=False):
        residual = self.conv2(self.conv1(x, reverse=reverse), reverse=reverse)
        x = residual + self.conv_residual(x, reverse=reverse)
        return x

class deconv_block(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(deconv_block, self).__init__()
        
        self.conv1 = QRNNDeConv3D(in_ch, out_ch, bn=bn)
        self.conv2 = QRNNDeConv3D(out_ch,out_ch, bn=bn)
        self.conv_residual = QRNNDeConv3D(in_ch, out_ch, k=1, s=1, p=0, bn=bn)

    def forward(self, x, reverse=False):
        residual = self.conv2(self.conv1(x, reverse=reverse), reverse=reverse)
        x = residual + self.conv_residual(x, reverse=reverse)
        return x

class U_Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, use_noise_map=False, bn=True):
        super(U_Net, self).__init__()
        self.use_2dconv = False
        self.bandwise = False
        self.use_noise_map = use_noise_map

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = QRNNConv3D(filters[0], filters[0], k=3, s=(1, 2, 2), p=1, bn=bn)
        self.Down2 = QRNNConv3D(filters[1], filters[1], k=3, s=(1, 2, 2), p=1, bn=bn)
        self.Down3 = QRNNConv3D(filters[2], filters[2], k=3, s=(1, 2, 2), p=1, bn=bn)
        self.Down4 = QRNNConv3D(filters[3], filters[3], k=3, s=(1, 2, 2), p=1, bn=bn)

        self.Conv1 = BiQRNNConv3D(in_ch, filters[0], bn=bn)
        self.Conv2 = conv_block(filters[0], filters[1], bn=bn)
        self.Conv3 = conv_block(filters[1], filters[2], bn=bn)
        self.Conv4 = conv_block(filters[2], filters[3], bn=bn)
        self.Conv5 = conv_block(filters[3], filters[4], bn=bn)

        self.Up5 = QRNNUpsampleConv3d(filters[4], filters[3], bn=bn)
        self.Up_conv5 = deconv_block(filters[4], filters[3], bn=bn)

        self.Up4 = QRNNUpsampleConv3d(filters[3], filters[2], bn=bn)
        self.Up_conv4 = deconv_block(filters[3], filters[2], bn=bn)

        self.Up3 = QRNNUpsampleConv3d(filters[2], filters[1], bn=bn)
        self.Up_conv3 = deconv_block(filters[2], filters[1], bn=bn)

        self.Up2 = QRNNUpsampleConv3d(filters[1], filters[0], bn=bn)
        self.Up_conv2 = deconv_block(filters[1], filters[0], bn=bn)

        self.Conv = BiQRNNDeConv3D(filters[0], 1, bias=True, bn=bn)

    def forward(self, x):
        # x: [B, C, B, W, H]
        e1 = self.Conv1(x)

        e2 = self.Down1(e1, reverse=True)
        e2 = self.Conv2(e2, reverse=False)

        e3 = self.Down2(e2, reverse=True)
        e3 = self.Conv3(e3, reverse=False)

        e4 = self.Down3(e3, reverse=True)
        e4 = self.Conv4(e4, reverse=False)

        e5 = self.Down4(e4, reverse=True)
        e5 = self.Conv5(e5, reverse=False)

        d5 = self.Up5(e5, reverse=True)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5, reverse=False)

        d4 = self.Up4(d5, reverse=True)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4, reverse=False)

        d3 = self.Up3(d4, reverse=True)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3, reverse=False)

        d2 = self.Up2(d3, reverse=True)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2, reverse=False)

        out = self.Conv(d2)

        if self.use_noise_map:
            return out + x[:,0,:,:,:].unsqueeze(1)
        else:
            return out + x