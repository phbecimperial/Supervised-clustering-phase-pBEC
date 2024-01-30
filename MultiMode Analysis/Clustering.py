import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("cnn.pt")
newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (
    0.5,))])  # Transforms it to a tensor, and rescales pixel values [0, 1]
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



with torch.no_grad():
    model.eval()

    all_features = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        all_features.append(outputs.cpu().numpy())

all_features = np.array(all_features)

#Insert PCA if needed
target_names=test_dataset.classes
kmeans = KMeans(n_clusters = len(target_names), n_jobs=-1)
kmeans.fit(all_features)

