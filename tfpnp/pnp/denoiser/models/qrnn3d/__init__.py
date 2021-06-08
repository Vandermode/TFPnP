from .memnet import MemNet
from .qrnn import REDC3D
from .qrnn import QRNNREDC3D
from .qrnn import ResQRNN3D
from .denet import DeNet
from .qrnn.unet import U_Net

"""Define commonly used architecture"""


def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def qrnn2d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True, is_2d=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def memnet():
    net = MemNet(31, 64, 6, 6)
    net.use_2dconv = True
    net.bandwise = False
    return net


def hsidenet():
    net = DeNet(in_channels=10)
    net.use_2dconv = True
    net.bandwise = False
    return net


def qrnn3d_masked():
    net = QRNNREDC3D(2, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def unet_masked():
    return U_Net(in_ch=2, out_ch=1, use_noise_map=True)

def unet_masked_nobn():
    return U_Net(in_ch=2, out_ch=1, use_noise_map=True, bn=False)

def unet():
    return U_Net(in_ch=1, out_ch=1, use_noise_map=False)

def unet_nobn():
    return U_Net(in_ch=1, out_ch=1, use_noise_map=False, bn=False)