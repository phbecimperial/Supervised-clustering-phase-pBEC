from pickle_Dataset import pickle_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


normalize = v2.Normalize(
mean=[0.4914],
std=[0.2023],
)

transform = v2.Compose([
        v2.Resize((224,224)),
        normalize,
])

dataset = pickle_Dataset(r'MultiMode Analysis/Training_Images', transforms=transform)
train_loader = DataLoader(dataset, 3, shuffle=True)

for images, key in train_loader:
    print(images, key)

print(dataset)