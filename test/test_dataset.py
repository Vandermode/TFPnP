import torch.utils.data
from tfpnp.data.dataset import HSIDeblurDataset

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
    
    
if __name__ == '__main__':
    test_hsi_dataset()