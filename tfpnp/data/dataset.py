import torch 
import torch.utils.data as data
 
class BaseDataset(data.dataset.Dataset):
    def __getitem__(self, index) -> dict:
        """ return a dict containing the required data
            must contains {'input': input data, 'output': initial output, 'gt': ground truth}
        """
        raise NotImplementedError
