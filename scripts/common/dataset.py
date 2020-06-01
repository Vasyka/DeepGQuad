import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SimpleSeqsDataset(Dataset):
    def __init__(self, seqs, labels=None, transforms=None):
        self.X = seqs
        self.y = labels
        self.transforms = transforms
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        data = np.asarray(data)
        
        data = torch.from_numpy(data).type(torch.FloatTensor).permute(1,0)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data