import cv2
import torch
import torch.nn as nn
from torch.nn import Module as M
from tqdm import tqdm as tqdm


batch_size = 1
class CNN(nn.Module):
    def __init__(self, classes, layer1=32, layer2=64, layer_k=3, pool_k=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, layer1, layer_k, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(pool_k, stride=2)


        self.conv2 = nn.Conv2d(layer1, layer2, layer_k, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(pool_k, stride=2, padding=1)

        self.flat = nn.Flatten()

        self.fc3 = nn.LazyLinear(128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flat(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

class CNN(nn.Module):
    def __init__(self, classes, layer1=32, layer2=64, layer_k=3, pool_k=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, layer1, layer_k, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(pool_k, stride=2)


        self.conv2 = nn.Conv2d(layer1, layer2, layer_k, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(pool_k, stride=2, padding=1)

        self.flat = nn.Flatten()

        self.fc3 = nn.LazyLinear(128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flat(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x






