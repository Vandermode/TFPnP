import torch
import torch.nn as nn
import torch.nn.functional as F

conv = nn.Conv2d


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None, bias=True):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, act=nn.LeakyReLU(0.2), num_layer=3):
        super(ConvBlock, self).__init__()
        self.add_module('conv-0', ConvLayer(
                conv, in_channels, channels, k, s, 
                padding=None, dilation=1, norm=None, act=act))

        for i in range(1, num_layer):
            self.add_module('conv-{}'.format(i), ConvLayer(
                conv, channels, channels, k, s, 
                padding=None, dilation=1, norm=None, act=act))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, requires_grad=False):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(512+256, 256)
        self.up2 = up(256+128, 128)
        self.up3 = up(128+64, 64)
        self.up4 = up(64+32, 32)
        self.outc = outconv(32, out_channels)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        noisy_img = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        residual = self.outc(x)
        
        C = residual.shape[1]
        return noisy_img[:, :C, ...] + residual


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


###################### SGN ######################
import torch
import torch.nn as nn
import math
import torch.nn.init as init

class ImageDownsample(nn.Conv2d):
    def __init__(self, n_colors, scale):
        super(ImageDownsample, self).__init__(n_colors, n_colors, kernel_size=2*scale, bias = False, padding=scale//2,stride=scale)
        kernel_size=2*scale
        self.weight.data = torch.zeros(n_colors,n_colors,kernel_size,kernel_size)
        for i in range(n_colors):
            self.weight.data[i,i,:,:] = torch.ones(1,1,kernel_size,kernel_size)/(kernel_size*kernel_size)
        self.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SimpleUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):
        m = []
        m.append(conv(n_feat, scale*scale*3, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleUpsampler, self).__init__(*m)

class SimpleGrayUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):
        m = []
        m.append(conv(n_feat, scale*scale, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleGrayUpsampler, self).__init__(*m)

def DownSamplingShuffle(x, scale=4):
    [N,C,W,H] = x.shape
    if(scale==4):
        x1 = x[:,:,0:W:4,0:H:4]
        x2 = x[:,:,0:W:4,1:H:4]
        x3 = x[:,:,0:W:4,2:H:4]
        x4 = x[:,:,0:W:4,3:H:4]
        x5 = x[:,:,1:W:4,0:H:4]
        x6 = x[:,:,1:W:4,1:H:4]
        x7 = x[:,:,1:W:4,2:H:4]
        x8 = x[:,:,1:W:4,3:H:4]
        x9 = x[:,:,2:W:4,0:H:4]
        x10 = x[:,:,2:W:4,1:H:4]
        x11 = x[:,:,2:W:4,2:H:4]
        x12 = x[:,:,2:W:4,3:H:4]
        x13 = x[:,:,3:W:4,0:H:4]
        x14 = x[:,:,3:W:4,1:H:4]
        x15 = x[:,:,3:W:4,2:H:4]
        x16 = x[:,:,3:W:4,3:H:4]
        return torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16),1)
    else:
        x1 = x[:,:,0:W:2,0:H:2]
        x2 = x[:,:,0:W:2,1:H:2]
        x3 = x[:,:,1:W:2,0:H:2]
        x4 = x[:,:,1:W:2,1:H:2]

        return torch.cat((x1,x2,x3,x4),1)


class Concate(nn.Module):
    def __init__(self):
        super(Concate, self).__init__()

    def forward(self, x, y):
        return torch.cat((x,y),1)


class Basic_Block(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size,bias=True, bn=False, act=nn.ReLU(True)):
        super(Basic_Block, self).__init__()
        m = []
        m.append(conv(in_feat,out_feat,kernel_size,bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_feat))
        if act is not None: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class SGNDN3(nn.Module):
    def __init__(self, in_channels, out_channels, g_blocks=3, m_blocks=2, n_feats=32, conv=default_conv):
        super(SGNDN3, self).__init__()
        kernel_size = 3 
        self.act = nn.LeakyReLU(0.2, inplace=False)
        # self.act = nn.ReLU(True)
        bn = False

        inputnumber = in_channels
        self.fusion = Concate() 
        self.upsampling = nn.PixelShuffle(2)

        m_lrhead1 = [conv(inputnumber*4, n_feats*2, 3)]
        m_lrbody1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody11 = [conv(n_feats*2, n_feats*2, 3)]

        m_lrhead2 = [conv(inputnumber*16, n_feats*4, 3)]
        m_lrbody2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody21 = [conv(n_feats*4, n_feats*4, 3)]

        m_lrhead3 = [conv(inputnumber*64, n_feats*8, 3)]
        m_lrbody3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody31 = [conv(n_feats*8, n_feats*8, 3)]
       
        # define head module
        m_head = [conv(in_channels, n_feats, 3)]

        # define body module
        m_lrtail1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act)]
        m_lrtail2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act)]
        m_lrtail3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act)]

        m_lrhead1_0 = [Basic_Block(conv, 3*n_feats, n_feats*2, kernel_size, bn=bn, act=self.act)]
        m_lrhead2_0 = [Basic_Block(conv, 6*n_feats, n_feats*4, kernel_size, bn=bn, act=self.act)]

        m_body0 = [Basic_Block(conv, int(1.5*n_feats),n_feats, kernel_size, bn=bn, act=self.act)  ]
        m_body1 = [Basic_Block(conv, n_feats, n_feats,  kernel_size, bn=bn, act=self.act) for _ in range(m_blocks)  ]

        m_tail = [conv(n_feats, out_channels, kernel_size)]

        self.lrhead1 = nn.Sequential(*m_lrhead1)
        self.lrbody1 = nn.Sequential(*m_lrbody1)
        self.lrtail1 = nn.Sequential(*m_lrtail1)
        self.lrhead2 = nn.Sequential(*m_lrhead2)
        self.lrbody2 = nn.Sequential(*m_lrbody2)
        self.lrtail2 = nn.Sequential(*m_lrtail2)
        self.lrhead3 = nn.Sequential(*m_lrhead3)
        self.lrbody3 = nn.Sequential(*m_lrbody3)
        self.lrtail3 = nn.Sequential(*m_lrtail3)	
	
        self.lrbody11 = nn.Sequential(*m_lrbody11)
        self.lrbody21 = nn.Sequential(*m_lrbody21)
        self.lrbody31 = nn.Sequential(*m_lrbody31)

        self.lrhead1_0 = nn.Sequential(*m_lrhead1_0)	
        self.lrhead2_0 = nn.Sequential(*m_lrhead2_0)	

        self.head = nn.Sequential(*m_head)
        self.body0 = nn.Sequential(*m_body0)
        self.body1 = nn.Sequential(*m_body1)
        self.tail = nn.Sequential(*m_tail)
        self.upsampling = nn.PixelShuffle(2)
        self.reset_params()

        m_upsampler = [SimpleUpsampler(conv, 1, n_feats)]
        self.upsampler = nn.Sequential(*m_upsampler)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        g1 = DownSamplingShuffle(x,2)
        g2 = DownSamplingShuffle(g1,2)
        g3 = DownSamplingShuffle(g2,2)

        g3 = self.act(self.lrhead3(g3))
        g3 = self.lrbody31(self.lrbody3(g3))+g3
        g3 = self.lrtail3(g3)
        g3 = self.upsampling(g3)

        g2 = self.act(self.lrhead2(g2))
        g2 = self.lrhead2_0(self.fusion(g2,g3))
        g2 = self.lrbody21(self.lrbody2(g2))+g2
        g2 = self.lrtail2(g2)
        g2 = self.upsampling(g2)

        g1 = self.act(self.lrhead1(g1))
        g1 = self.lrhead1_0(self.fusion(g1,g2))
        g1 = self.lrbody11(self.lrbody1(g1))+g1
        g1 = self.lrtail1(g1)		
        g1 = self.upsampling(g1)
		
        residual = self.head(x)
        residual = self.fusion(g1,self.act(residual))
        residual = self.body0(residual)
        residual = self.body1(residual)

        out = self.tail(residual)

        C = out.shape[1]
        out = out + x[:, :C, ...]
        return out


