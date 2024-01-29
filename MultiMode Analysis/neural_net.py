import cv2
import torch
import torch.nn as nn
from torch.nn import Module as M
from tqdm import tqdm as tqdm


batch_size = 1
class CNN(nn.Module):
    def __init__(self, classes=10, layer1=32, layer2=64, layer_k=3, pool_k=2):
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



epochs=5

for epochs in tqdm(range(epochs)):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in tqdm(range(epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print training loss for each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")





