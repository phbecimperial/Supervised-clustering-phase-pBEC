import os
import lzma
import pickle
import numpy as np
from glob import glob
import torch
from torch.utils import data


class pickle_Dataset(data.Dataset):
    def __init__(self,  root = '.', iext = '.pt', transforms=None, train = False):
        self.transforms = transforms

        self.dir_list = np.array(glob(root + r'\*'))
        self.size = len(self.dir_list)
        
    def __getitem__(self, index):

        path = self.dir_list[index]

        with lzma.open(path, 'rb') as f:
            im, key = pickle.load(f)
            im = self.transforms(im)

        return (im, key)

    def __len__(self):
        return self.size 