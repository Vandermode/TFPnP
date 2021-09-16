"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch_radon import Radon


class Radon_norm(Radon):
    def __init__(self, resolution, angles, det_count=-1, det_spacing=1.0, clip_to_circle=False, opnorm=None):
        super(Radon_norm, self).__init__(resolution, angles,
                                         det_count, det_spacing, clip_to_circle)
        if opnorm is None:
            def normal_op(x): return super(Radon_norm, self).backward(
                super(Radon_norm, self).forward(x))
            x = torch.randn(1, 1, resolution, resolution).cuda()
            opnorm = power_method_opnorm(normal_op, x, n_iter=10)
        self.opnorm = opnorm
        self.resolution = resolution
        self.view = angles.shape[0]

    def backprojection_norm(self, sinogram):
        return self.backprojection(sinogram) / self.opnorm**2

    def filter_backprojection(self, sinogram):
        sinogram = self.filter_sinogram(sinogram, filter_name='ramp')
        return self.backprojection(sinogram)

    def normal_operator(self, x):
        return self.backprojection_norm(self.forward(x))


def create_radon(resolution, view, opnorm):
    angles = torch.linspace(0, 179/180*np.pi, view)
    det_count = int(np.ceil(np.sqrt(2) * resolution))
    radon = Radon_norm(resolution, angles, det_count, opnorm=opnorm)
    return radon


class RadonGenerator:
    def __init__(self):
        self.opnorms = {}

    def __call__(self, resolution, view):
        key = (resolution, view)
        if key in self.opnorms:
            opnorm = self.opnorms[key]
            radon = create_radon(resolution, view, opnorm)
        else:
            radon = create_radon(resolution, view, opnorm=None)
            opnorm = radon.opnorm
            self.opnorms[key] = opnorm

        return radon


def power_method_opnorm(normal_op, x, n_iter=10):
    def _normalize(x):
        size = x.size()
        x = x.view(size[0], -1)
        norm = torch.norm(x, dim=1)
        x /= norm.view(-1, 1)
        return x.view(*size), torch.max(norm).item()

    with torch.no_grad():
        x, _ = _normalize(x)

        for i in range(n_iter):
            next_x = normal_op(x)
            x, v = _normalize(next_x)

    return v**0.5


def real2complex(x):
    return torch.stack([x, torch.zeros_like(x)], dim=4)


def complex2real(x):
    return x[..., 0]


def complex2channel(x):
    N, C, H, W, _ = x.shape
    # N C H W 2 -> N 2C H W
    temp = x
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    x = x.view(N, C*2, H, W)
    return x


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_mul(x1, x2):
    """
    Compute multiplication of two complex numbers
    """
    assert x1.size(-1) == 2 and x2.size(-1) == 2

    res = torch.stack(
        (x1[..., 0]*x2[..., 0]-x1[..., 1]*x2[..., 1],
         x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]), -1)

    return res


def conjugate(x):
    return torch.stack([x[..., 0], -x[..., 1]], -1)


def cdp_forward(data, mask):
    """
    Compute the forward model of cdp.

    Args:
        data (torch.Tensor): Image_data (batch_size*1*hight*weight*2).
        mask (torch.Tensor): mask (batch_size*sampling_rate*hight*weight*2), where the size of the final dimension
            should be 2 (complex value).

    Returns:
        forward_data (torch.Tensor): the complex field of forward data (batch_size*sampling_rate*hight*weight*2)
    """
    assert mask.size(-1) == 2
    if data.ndimension() == 4:
        data = torch.stack([data, torch.zeros_like(data)], -1)
    sampling_rate = mask.shape[1]
    x = data.repeat(1, sampling_rate, 1, 1, 1)
    masked_data = complex_mul(x, mask)
    forward_data = torch.fft(masked_data, 2, normalized=True)
    return forward_data


def cdp_backward(data, mask):
    """
    Compute the backward model of cdp (the inverse operator of forward model).

    Args:
        data (torch.Tensor): Field_data (batch_size*sampling_rate*hight*weight*2).
        mask (torch.Tensor): mask (batch_size*sampling_rate*hight*weight*2), where the size of the final dimension
            should be 2 (complex value).

    Returns:
        backward_data (torch.Tensor): the complex field of backward data (batch_size*1*hight*weight*2)
    """
    assert mask.size(-1) == 2
    sampling_rate = mask.shape[1]
    Ifft_data = torch.ifft(data, 2, normalized=True)
    backward_data = complex_mul(Ifft_data, conjugate(mask))
    return backward_data.mean(1, keepdim=True)


if __name__ == '__main__':
    from scipy.io import loadmat
    from os.path import join

    maskdir = '/media/kaixuan/DATA/Papers/Code/Data/MRI/masks'
    sampling_masks = ['radial_128_2', 'radial_128_4',
                      'radial_128_8']  # different masks
    obs_masks = [loadmat(join(maskdir, '{}.mat'.format(sampling_mask))).get(
        'mask') for sampling_mask in sampling_masks]
    mask = torch.from_numpy(obs_masks[2])[None].bool()

    def csmri_normal_op(x):
        y0 = fft2(x)
        y0[:, ~mask, :] = 0
        ATy0 = ifft2(y0)
        return ATy0

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 1, 128, 128, 2), device=device)

    opnorm = power_method_opnorm(csmri_normal_op, x, 10)
    print(opnorm)  # nearly equal to 1
