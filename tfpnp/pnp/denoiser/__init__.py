import os

import torch

from .models.unet import UNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Denoiser:
    def denoise(self, x, sigma):
        """ Denoise x with noise level sigma

        Args:
            x (torch.tensor): input image with shape [B,C,H,W]
            sigma (torch.tensor): noise level with shape [B].

        Return:
            out (torch.tensor): denoised image with the same shape of the input
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the denoiser to the required device
        """
        raise NotImplementedError


class UNetDenoiser2D(Denoiser):
    def __init__(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(CURRENT_DIR, 'pretrained', 'unet-nm.pt')

        net = UNet(2, 1)
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        self.net = net

    def denoise(self, x, sigma):
        # x: [B,1,H,W]
        N, C, H, W = x.shape

        sigma = sigma.view(N, 1, 1, 1)

        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma
        out = self.net(torch.cat([x, noise_map], dim=1))

        return torch.clamp(out, 0, 1)

    def to(self, device):
        self.net.to(device)
        return self


class GRUNetDenoiser:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                CURRENT_DIR, 'pretrained', 'grunet-unet-qrnn3d.pth')

        from .models.qrnn3d import unet_masked_nobn
        model = unet_masked_nobn()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False

        self.model = model

    def denoise(self, x, sigma):
        # x: [B,C,H,W]
        x = torch.unsqueeze(x, 1)
        s = torch.ones((x.shape[0], 1, x.shape[2],
                        x.shape[3], x.shape[4])).to(x.device)
        s *= sigma.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat((x, s), dim=1)

        x = self.model(x)
        x = torch.squeeze(x)
        if x.ndim == 3:
            x = torch.unsqueeze(x, 0)
        return torch.clamp(x, 0, 1)

    def to(self, device):
        self.model.to(device)
        return self
