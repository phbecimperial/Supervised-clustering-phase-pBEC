import os
import lzma
import pickle
import numpy as np
from glob import glob
import torch
from torch.utils import data
from torchvision.transforms import v2


class pickle_Dataset(data.Dataset):
    def __init__(self,  root = '.', transforms=None, max = np.inf):
        self.transforms = transforms

        self.dir_list = np.array(glob(root + r'\*'))

        if len(self.dir_list) > max:
            self.dir_list = self.dir_list[:max]
        self.size = len(self.dir_list)
        
    def __getitem__(self, index):

        path = self.dir_list[index]

        with lzma.open(path, 'rb') as f:
            im, key = pickle.load(f)
            im = np.asarray(im, np.float32)
            im = self.transforms(im)

        return (im, torch.from_numpy(np.array(key, np.float32)))

    def __len__(self):
        return self.size 