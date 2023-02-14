import numpy as np
import torch

torch_fft = torch.fft
torch_ifft = torch.ifft 

# ---------------------------------------------------------------------------- #
#                                   CSMRI                                      #
# ---------------------------------------------------------------------------- #


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
    data = torch_fft(data, 2, normalized=True)
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
    data = torch_ifft(data, 2, normalized=True)
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
        (x1[..., 0]*x2[..., 0] - x1[..., 1]*x2[..., 1],
         x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]), -1)

    return res


def conjugate(x):
    return torch.stack([x[..., 0], -x[..., 1]], -1)


# ---------------------------------------------------------------------------- #
#                                     PR                                       #
# ---------------------------------------------------------------------------- #


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


def cpr_forward(data, mask, sample_matrix):
    """
    Compute the forward model of compressive phase retrieval.

    Args:
        data (torch.Tensor): Image_data (batch_size*1*hight*weight*2).
        mask (torch.Tensor): mask (batch_size*1*hight*weight*2), where the size of the final dimension
            should be 2 (complex value).
        sample_matrix (torch.Tensor): undersampling matrix (m*n), n = hight*weight, m = samplingratio*n

    Returns:
        forward_data (torch.Tensor): the complex field of forward data (batch_size*1*m*2)
    """    
    assert mask.size(-1) == 2
    if data.ndimension() == 4:
        data = torch.stack([data, torch.zeros_like(data)], -1)    
    B, _, H, W, _ = data.shape
    m, n = sample_matrix.shape
    masked_data = complex_mul(data, mask)    
    fourier_data = torch.fft(masked_data, 2, normalized=True).view(B, 1, H*W, 2)
    forward_data = torch.einsum('bcnk,mn->bcmk', fourier_data, sample_matrix) * (n/m)**0.5
    return forward_data


def cpr_backward(data, mask, sample_matrix):
    """
    Compute the backward model of cpr (the inverse operator of forward model).

    Args:
        data (torch.Tensor): Field_data (batch_size*1*m*2).
        mask (torch.Tensor): mask (batch_size*1*hight*width*2), where the size of the final dimension
            should be 2 (complex value).
        sample_matrix (torch.Tensor): undersampling matrix (m*n).

    Returns:
        backward_data (torch.Tensor): the complex field of backward data (batch_size*1*hight*weight*2)
    """
    assert mask.size(-1) == 2
    B = data.shape[0]
    _, _, H, W, _ = mask.shape
    m, n = sample_matrix.shape
    back_data = torch.einsum('bcmk,mn->bcnk', data, sample_matrix)
    # print(back_data.sum())
    # print(back_data[0, 0, 0:4, 0])    
    Ifft_data = torch.ifft(back_data.view(B, 1, H, W, 2), 2, normalized=True)
    # print(Ifft_data.sum())
    # print(Ifft_data[0, 0, 0:4, 0:4, 0])
    backward_data = complex_mul(Ifft_data, conjugate(mask)) * (n/m)**0.5    
    
    return backward_data


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]

    return res.reshape(siz0 + siz1)


# ---------------------------------------------------------------------------- #
#                                     SPI                                      #
# ---------------------------------------------------------------------------- #


def spi_forward(x, K, alpha, q):
    ones = torch.ones(1, 1, K, K).to(x.device)
    theta = alpha * kron(x, ones) / (K**2)
    y = torch.poisson(theta)
    ob = (y >= torch.ones_like(y) * q).float()

    return ob


def spi_inverse(ztilde, K1, K, mu):
    """
    Proximal operator "Prox\_{\frac{1}{\mu} D}" for single photon imaging
    assert alpha == K and q == 1
    """
    z = torch.zeros_like(ztilde)

    K0 = K**2 - K1
    indices_0 = (K1 == 0)

    z[indices_0] = ztilde[indices_0] - (K0 / mu)[indices_0]

    func = lambda y: K1 / (torch.exp(y) - 1) - mu * y - K0 + mu * ztilde

    indices_1 = torch.logical_not(indices_0)

    # differentiable binary search
    bmin = 1e-5 * torch.ones_like(ztilde)
    bmax = 1.1 * torch.ones_like(ztilde)

    bave = (bmin + bmax) / 2.0

    for i in range(10):
        tmp = func(bave)
        indices_pos = torch.logical_and(tmp > 0, indices_1)
        indices_neg = torch.logical_and(tmp < 0, indices_1)
        indices_zero = torch.logical_and(tmp == 0, indices_1)
        indices_0 = torch.logical_or(indices_0, indices_zero)
        indices_1 = torch.logical_not(indices_0)

        bmin[indices_pos] = bave[indices_pos]
        bmax[indices_neg] = bave[indices_neg]
        bave[indices_1] = (bmin[indices_1] + bmax[indices_1]) / 2.0

    z[K1 != 0] = bave[K1 != 0]
    return torch.clamp(z, 0.0, 1.0)


# ---------------------------------------------------------------------------- #
#                                     CT                                       #
# ---------------------------------------------------------------------------- #
try:
    from torch_radon import Radon
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


    class Radon_norm(Radon):
        def __init__(self, resolution, angles, det_count=-1, det_spacing=1.0, clip_to_circle=False, opnorm=None):
            super(Radon_norm, self).__init__(resolution, angles, det_count, det_spacing, clip_to_circle)
            if opnorm is None:
                normal_op = lambda x: super(Radon_norm, self).backward(super(Radon_norm, self).forward(x))
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
except:
    pass

if __name__ == '__main__':
    # from scipy.io import loadmat
    # from os.path import join

    # maskdir = '/media/kaixuan/DATA/Papers/Code/Data/MRI/masks'
    # sampling_masks = ['radial_128_2', 'radial_128_4',
    #                   'radial_128_8']  # different masks
    # obs_masks = [loadmat(join(maskdir, '{}.mat'.format(sampling_mask))).get(
    #     'mask') for sampling_mask in sampling_masks]
    # mask = torch.from_numpy(obs_masks[2])[None].bool()

    # def csmri_normal_op(x):
    #     y0 = fft2(x)
    #     y0[:, ~mask, :] = 0
    #     ATy0 = ifft2(y0)
    #     return ATy0

    # # device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn((1, 1, 128, 128, 2), device=device)

    # opnorm = power_method_opnorm(csmri_normal_op, x, 10)
    # print(opnorm)  # nearly equal to 1

    ######################
    
    from scipy.io import loadmat
    from pathlib import Path
    
    basedir = Path('/home/kaixuan/code/Data/PR/pr')
    data = loadmat(basedir / 'data/Test_prdeep_128.mat')['labels']
    mask = loadmat(basedir / 'sampling_matrix/mask_0_128.mat')['Mask']
    sample_matrix = loadmat(basedir / 'sampling_matrix/SampM_30_128.mat')['SubsampM']
    
    data = torch.from_numpy(data).float().unsqueeze_(1)
    mask = torch.from_numpy(mask)
    mask = torch.stack([mask.real, mask.imag], -1).float()[None][None]
    sample_matrix = torch.from_numpy(sample_matrix).float()
        
    ob = cpr_forward(data, mask, sample_matrix)    
    print(ob[0, 0, 0:10, 0])
    backward_data = cpr_backward(ob, mask, sample_matrix)
    print(backward_data[0, 0, :4, :4, 0])
    
    print(ob.sum())
    print(backward_data.sum())
    
    pass
