import torch 
import torch.utils.data as data
import os

class BaseDataset(data.dataset.Dataset):
    def __getitem__(self, index) -> dict:
        """ return a dict containing the required data
            must contains {'input': input data, 'output': initial output, 'gt': ground truth}
        """
        raise NotImplementedError

class ImageDir(BaseDataset):
    def __init__(self, datadir, training=True):
        self.datadir = datadir
        self.fns = [im for im in os.listdir(self.datadir) if im.endswith(".mat")]  
        self.fns = self.fns[10:] if training else self.fns[:10]
        
    def __getitem__(self, index) -> dict:
        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        data = self._get_data(imgpath)
        return data
        
    def _get_data(self, path):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.fns)
    