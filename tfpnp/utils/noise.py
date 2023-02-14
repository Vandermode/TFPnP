import numpy as np
import torch


class GaussianModelC:  # continuous noise levels
    def __init__(self, low_sigma=0, high_sigma=55):
        super().__init__()
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        
    def __call__(self, x):
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        
        sigma = sigma / 255.
        y = x + torch.randn(*x.shape) * sigma

        return y, sigma


class GaussianModelD:  # discrete noise levels
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas
        
    def __call__(self, x, idx=None):
        if idx is not None:
            sigma = self.sigmas[idx]
        else:
            sigma = np.random.choice(self.sigmas) # random is important
        sigma = sigma / 255.
        y = x + torch.randn(*x.shape) * sigma
              
        return y, sigma


class GaussianModelP:  # noise percentages
    def __init__(self, sigmas_p, batch_mode=False):
        super().__init__()
        self.sigmas_p = sigmas_p
        self.batch_mode = batch_mode
        
    def __call__(self, x):
        if not self.batch_mode:
            sigma = np.random.choice(self.sigmas_p).astype(np.float32)
            y = x + torch.randn_like(x) * torch.mean(torch.abs(x)) * sigma
        else:
            N = x.shape[0]
            sigma = np.random.choice(self.sigmas_p, size=N)
            sigma = torch.from_numpy(sigma).view(N, 1, 1, 1).float().to(x.device)
            x_mean = torch.mean(torch.abs(x).view(N, -1), dim=1).view(N, 1, 1, 1)
            y = x + torch.randn_like(x) * x_mean * sigma       

        return y.float(), sigma


class PoissonModel: # for phase retrieval 
    def __init__(self, alphas):
        super().__init__()
        self.alphas = alphas
        
    def __call__(self, z, idx=None):
        if idx is not None:
            alpha = self.alphas[idx]
        else:
            alpha = np.random.choice(self.alphas)
        
        z2 = z ** 2
        intensity_noise = alpha/255 * torch.abs(z) * torch.randn_like(z)
                
        y2 = torch.clamp(z2 + intensity_noise, min=0)
        y = torch.sqrt(y2)

        rr =  y - torch.abs(z)
        sigma = rr.std()

        return y, sigma
