from tfpnp.pnp.denoiser import GRUNetDenoiser, UNetDenoiser2D
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_grunet_denoiser():  
    denoiser = GRUNetDenoiser('model/grunet-unet-qrnn3d.pth').to(device)
    
    gt = torch.ones((16, 31, 128, 128)).to(device)
    sigma = torch.ones(gt.shape[0]).to(device)
    out = denoiser.denoise(gt, sigma)
    
    print(gt.shape)
    print(sigma.shape)
    print(out.shape)
    
    # torch.Size([16, 31, 128, 128])
    # torch.Size([16])
    # torch.Size([16, 31, 128, 128])

def test_unet2d_denoiser():
    denoiser = UNetDenoiser2D('model/unet-nm.pt').to(device)
    
    gt = torch.ones((16, 1, 128, 128)).to(device)
    sigma = torch.ones(gt.shape[0]).to(device)
    out = denoiser.denoise(gt, sigma)
    
    print(gt.shape)
    print(sigma.shape)
    print(out.shape)

if __name__ == '__main__':
    # test_grunet_denoiser()
    test_unet2d_denoiser()