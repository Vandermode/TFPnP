import torch
import torch.nn as nn


class QRNNREDC3D(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx,
                 BiQRNNConv3D=None, BiQRNNDeConv3D=None,
                 QRNN3DEncoder=None, QRNN3DDecoder=None, is_2d=False, has_ad=True, bn=True, act='tanh', plain=False):
        super(QRNNREDC3D, self).__init__()
        assert sample_idx is None or isinstance(sample_idx, list)

        self.enable_ad = has_ad
        if sample_idx is None: sample_idx = []
        if is_2d:
            self.feature_extractor = BiQRNNConv3D(in_channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act)
        else:
            self.feature_extractor = BiQRNNConv3D(in_channels, channels, bn=bn, act=act)

        self.encoder = QRNN3DEncoder(channels, num_half_layer, sample_idx, is_2d=is_2d, has_ad=has_ad, bn=bn, act=act,
                                     plain=plain)
        self.decoder = QRNN3DDecoder(channels * (2 ** len(sample_idx)), num_half_layer, sample_idx, is_2d=is_2d,
                                     has_ad=has_ad, bn=bn, act=act, plain=plain)

        if act == 'relu':
            act = 'none'

        if is_2d:
            self.reconstructor = BiQRNNDeConv3D(channels, 1, bias=True, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn,
                                                act=act)
        else:
            self.reconstructor = BiQRNNDeConv3D(channels, 1, bias=True, bn=bn, act=act)

    def forward(self, x):
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        if self.enable_ad:
            out, reverse = self.encoder(out, xs, reverse=False)
            out = self.decoder(out, xs, reverse=(reverse))
        else:
            out = self.encoder(out, xs)
            out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.reconstructor(out)
        out = out + xs.pop()[:, 0, :].unsqueeze(1)
        return out


class QRNN3DEncoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, QRNNConv3D=None,
                 is_2d=False, has_ad=True, bn=True, act='tanh', plain=False):
        super(QRNN3DEncoder, self).__init__()
        # Encoder        
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in range(num_half_layer):
            if i not in sample_idx:
                if is_2d:
                    encoder_layer = QRNNConv3D(channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act)
                else:
                    encoder_layer = QRNNConv3D(channels, channels, bn=bn, act=act)
            else:
                if is_2d:
                    encoder_layer = QRNNConv3D(channels, 2 * channels, k=(1, 3, 3), s=(1, 2, 2), p=(0, 1, 1), bn=bn,
                                               act=act)
                else:
                    if not plain:
                        encoder_layer = QRNNConv3D(channels, 2 * channels, k=3, s=(1, 2, 2), p=1, bn=bn, act=act)
                    else:
                        encoder_layer = QRNNConv3D(channels, 2 * channels, k=3, s=(1, 1, 1), p=1, bn=bn, act=act)

                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x = self.layers[i](x)
                xs.append(x)
            x = self.layers[-1](x)

            return x
        else:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x = self.layers[i](x, reverse=reverse)
                reverse = not reverse
                xs.append(x)
            x = self.layers[-1](x, reverse=reverse)
            reverse = not reverse

            return x, reverse


class QRNN3DDecoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, QRNNDeConv3D=None, QRNNUpsampleConv3d=None,
                 is_2d=False, has_ad=True, bn=True, act='tanh', plain=False):
        super(QRNN3DDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                if is_2d:
                    decoder_layer = QRNNDeConv3D(channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act)
                else:
                    decoder_layer = QRNNDeConv3D(channels, channels, bn=bn, act=act)
            else:
                if is_2d:
                    decoder_layer = QRNNUpsampleConv3d(channels, channels // 2, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn,
                                                       act=act)
                else:
                    if not plain:
                        decoder_layer = QRNNUpsampleConv3d(channels, channels // 2, bn=bn, act=act)
                    else:
                        decoder_layer = QRNNDeConv3D(channels, channels // 2, bn=bn, act=act)

                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            x = self.layers[0](x)
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x)
            return x
        else:
            num_half_layer = len(self.layers)
            x = self.layers[0](x, reverse=reverse)
            reverse = not reverse
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x, reverse=reverse)
                reverse = not reverse
            return x
