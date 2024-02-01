import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neural_net import CNN
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
import time
import numpy as np
from custom_dataset import CustomDataset
from resnet import ResNet, ResidualBlock

import argparse

# Testing with MNIST first!

epochs = 100
classes = 14  # Key parameter
batch_size = 64
learning_rate = 0.001
val_split = 0.15
test_split = 0.15


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


early_stopper = EarlyStopper(patience=3, min_delta=1)  # Need to find best min_delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset for MNIST tests
# Transforms it to a tensor, and rescales pixel values [0, 1]

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#Load training data for training tests
new_size = (224, 224)
train_datasetini = CustomDataset(root_dir='Training_images',
                              new_size=new_size)  # Increase size/decrease stride + kernel inside resnet to increase acc

numTrainSamp = int(len(train_datasetini)) * (1 - val_split - test_split)
numValSamp = int(len(train_datasetini)) * val_split
numTestSamp = int(len(train_datasetini)) * test_split

# 3 sets of data: training, validation, testing, split
(train_dataset, validate_dataset, test_dataset) = torch.utils.data.random_split(train_datasetini,
                                                                                [int(numTrainSamp), int(numValSamp),
                                                                                 int(numTestSamp)],
                                                                                generator=None)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
# model = CNN(classes)
model = ResNet(ResidualBlock, layers=[3, 4, 6, 3], kernel_size=7, strides=2).to(
    device)  # ResNet 18,
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

trainSteps = len(train_loader.dataset) // batch_size
valSteps = len(val_loader.dataset) // batch_size

start = time.time()

# Training loop

for epoch in tqdm(range(epochs)):
    model.train()

    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update losses
        totalTrainLoss += loss
        trainCorrect += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            totalValLoss += loss
            valCorrect += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    #if early_stopper.early_stop(totalValLoss):
    #    break

    trainCorrect = trainCorrect / len(train_loader.dataset)
    valCorrect = valCorrect / len(val_loader.dataset)

    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    history["train_acc"].append(trainCorrect)
    history["val_loss"].append(avgValLoss.cpu().detach().numpy())
    history["val_acc"].append(valCorrect)

    accuracy = correct / total
    print(f"Test Accuracy Epoch {epoch + 1}/{epochs}: {accuracy * 100:.2f}%")

    end = time.time()
    print("Time taken to train", np.round(end - start))

# Now, use test dataset:

with torch.no_grad():
    model.eval()

    preds = []
    actual = np.array([])

    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        preds.extend(outputs.argmax(axis=1).cpu().numpy())
        actual = np.append(actual, labels.cpu().numpy())


print(classification_report(actual, np.array(preds), target_names=train_datasetini.classes))

import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="trainloss")
plt.plot(history["val_loss"], label="val_loss")
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

torch.save(model, 'Saved_models/Model_4_ResNet')
print('Saved_models/Model_4_ResNet')
