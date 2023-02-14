import os
import torch
from .models.unet import UNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class UNetDenoiser2D(torch.nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = os.path.join(CURRENT_DIR, 'pretrained', 'unet-nm.pt')
            if not os.path.exists(ckpt_path):
                raise ValueError('Default ckpt not found, you have to provide a ckpt path')
            
        net = UNet(2, 1)
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        self.net = net

    def forward(self, x, sigma):
        # x: [B,1,H,W]
        N, C, H, W = x.shape

        sigma = sigma.view(N, 1, 1, 1)

        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma
        out = self.net(torch.cat([x, noise_map], dim=1))

        return torch.clamp(out, 0, 1)
