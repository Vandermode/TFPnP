import os
import torch.utils.data
from scipy.io import loadmat

from tfpnp.data.dataset import HSIDeblurDataset
from tfpnp.data.dataset import CSMRIDataset, CSMRIEvalDataset
from tfpnp.data.noise_models import GaussianModelD


# export PYTHONPATH="/home/laizeqiang/Desktop/lzq/projects/tfpnp/tfpnp2/"
# python test/test_dataloader.py

def test_hsi_dataset():
    train_dataset =  HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    iterator = iter(train_loader)
    data = iterator.__next__()
    print(data.keys())
    for k, v in data.items():
        print(k, v.shape)
        
    # dict_keys(['low', 'FB', 'FBC', 'F2B', 'FBFy', 'gt'])
    # low torch.Size([2, 31, 512, 512])
    # FB torch.Size([2, 1, 512, 512, 2])
    # FBC torch.Size([2, 1, 512, 512, 2])
    # F2B torch.Size([2, 1, 512, 512, 2])
    # FBFy torch.Size([2, 31, 512, 512, 2])
    # gt torch.Size([2, 31, 512, 512])


def test_csmri_dataset():
    train_dir = 'data/'
    mri_dir = 'data/'
    mask_dir = 'data/masks'
    
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']  # different masks
    train_root = os.path.join(train_dir, 'Medical_128')
    val_roots = [os.path.join(mri_dir, 'Medical7_2020', sampling_mask, '15') for sampling_mask in sampling_masks]
    obs_masks = [loadmat(os.path.join(mask_dir, '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks]
    
    train_dataset = CSMRIDataset(train_root, fns=None, masks=obs_masks, noise_model=noise_model, repeat=12*100)
    val_datasets = [CSMRIEvalDataset(val_root, fns=None) for val_root in val_roots]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    iterator = iter(train_loader)
    data = iterator.__next__()
    print(data.keys())
    for k, v in data.items():
        print(k, v.shape)
        
    iterator = iter(val_loaders[0])
    data = iterator.__next__()
    print(data.keys())
    for k, v in data.items():
        try:
            print(k, v.shape)
        except:
            print(k, v)

    # dict_keys(['y0', 'x0', 'ATy0', 'gt', 'mask', 'sigma_n', 'output'])
    # y0 torch.Size([2, 1, 128, 128, 2])
    # x0 torch.Size([2, 1, 128, 128, 2])
    # ATy0 torch.Size([2, 1, 128, 128, 2])
    # gt torch.Size([2, 1, 128, 128])
    # mask torch.Size([2, 128, 128])
    # sigma_n torch.Size([2, 1, 128, 128, 2])
    # output torch.Size([2, 1, 128, 128])
    
    # dict_keys(['y0', 'ATy0', 'gt', 'name', 'x0', 'sigma_n', 'mask'])
    # y0 torch.Size([1, 1, 128, 128, 2])
    # ATy0 torch.Size([1, 1, 128, 128, 2])
    # gt torch.Size([1, 1, 128, 128])
    # name ['Brain_data1']
    # x0 torch.Size([1, 1, 128, 128, 2])
    # sigma_n torch.Size([1, 1, 128, 128, 2])
    # mask torch.Size([1, 128, 128])
    
if __name__ == '__main__':
    # test_hsi_dataset()
    test_csmri_dataset()