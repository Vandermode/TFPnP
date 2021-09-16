import torch
import torch.fft

def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))
    STy = upsample(x, sf=sf)
    FBFy = cmul(FBC, torch.rfft(STy, 3, onesided=False))
    return FB, FBC, F2B, FBFy


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-1,-2))
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf.imag[torch.abs(otf.imag) <  n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c

def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)

def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)
