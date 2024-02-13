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


from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle
import lzma
import torch
from modes import names
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir, new_size=(28, 28)):
        """
        Will get image file and unpickle. You should transform to tensor.

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(new_size, antialias=True),  transforms.Normalize((0.5,), (
                0.5,))])
        self.targets = []
        for f in self.file_list:
            parts = f.split('@')
            parts = parts[2].split('.')
            self.targets.append(parts[0])

        self.classes = names


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = lzma.open(img_name,'rb')
        image, label = pickle.load(image)
        image = self.transform(image)
        image = image.to(torch.float32)

        label = torch.tensor(label, dtype = torch.long)
        return image, label