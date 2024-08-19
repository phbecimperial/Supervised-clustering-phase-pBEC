import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mode_classifier import ResNet, ResidualBlock
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
import time
import numpy as np
import argparse
from pickle_Dataset import pickle_Dataset
from Ensamble_Classifier import Ensamble
import gc
# Testing with MNIST first!

epochs = 5
classes = 10  # 10 numbers
batch_size = 16
learning_rate = 0.01
train_split = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize((0.5,), (
    0.5,))])  # Transforms it to a tensor, and rescales pixel values [0, 1]
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = pickle_Dataset(root = r'Training_Images', transforms = transform, max = 2500)
test_dataset = pickle_Dataset(root = r'Training_Images', transforms = transform, max = 2500)

print(len(train_dataset))

numTrainSamp = int(len(train_dataset)) * train_split
numValSamp = int(len(train_dataset)) * (1 - train_split)

(train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)],
                                                                  generator=None)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
criterion = nn.CrossEntropyLoss()

model = Ensamble(ResNet, 13, 0.01, optim.Adam, criterion, device, block = ResidualBlock, layers = [3, 4, 6, 3])


history = {
    "train_loss": [],
    "train_error": [],
    "val_loss": [],
    "val_error": []
}

trainSteps = len(train_loader.dataset) // batch_size
valSteps = len(val_loader.dataset) // batch_size

start = time.time()

# 3 sets of data: training, validation, testing


# Training loop
epochs = 100
for epoch in tqdm(range(epochs)):
    model.train()

    train_rms_errors = []
    train_losses = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        model.loss_step(outputs, labels)

        loss = model.calculate_losses(outputs, labels)

        # Update losses
        train_losses.append(loss.numpy(force = True))
        
        torch.cuda.empty_cache()
        gc.collect()

        #train_rms_errors.append(rms_error)
        del inputs, labels, outputs


    history["train_error"].append(np.mean(train_rms_errors))
    history["train_loss"].append(np.mean(train_losses))



    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {history['train_loss'][epoch]}")
    print(f'mean RMS_error training for epoch = {history["train_error"][epoch]}')

    val_rms_errors = []
    val_losses = []
    #validation
    model.eval()
    with torch.no_grad():

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            rms_error = np.sqrt(np.mean((outputs - labels).numpy(force = True))**2)
            val_rms_errors.append(rms_error)
            val_losses.append(loss.numpy(force = True))
            del inputs, labels, outputs
        
        history["val_error"].append(np.mean(val_rms_errors))
        history["val_loss"].append(np.mean(val_losses))




torch.save(model, r"MultMode Analysis\Models\test_res.pt")
with open(r"MultMode Analysis\Models\test_res_hist.pkl", 'wb') as f:
    pkl.dump(history, f)


end = time.time()
print("Time taken to train", np.round(end - start))


import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="trainloss")
plt.plot(history["val_loss"], label="val_loss")
plt.plot(history["train_error"], label="train_acc")
plt.plot(history["val_error"], label="val_acc")

plt.title("Training Loss and Error")
plt.xlabel("Epoch")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

torch.save(model, r"MultMode Analysis\Models\test_res.pt")
