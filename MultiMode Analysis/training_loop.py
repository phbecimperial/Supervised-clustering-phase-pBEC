import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neural_net as net
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
import time
import numpy as np

#Testing with MNIST first!

epochs=5
classes=10 #10 numbers
batch_size=64
learning_rate=0.001
train_split = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) #Transforms it to a tensor, and rescales pixel values [0, 1]
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

numTrainSamp = int(len(train_dataset))*train_split
numValSamp = int(len(train_dataset))*(1-train_split)

(train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)], generator=None)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = net.CNN(classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

trainSteps = len(train_loader.dataset)//batch_size
valSteps = len(val_loader.dataset)//batch_size


start = time.time()

#3 sets of data: training, validation, testing


# Training loop
epochs = 5
for epoch in tqdm(range(epochs)):
    model.train()

    totalTrainLoss=0
    totalValLoss=0
    trainCorrect=0
    valCorrect=0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #Update losses
        totalTrainLoss+=loss
        trainCorrect+=(outputs.argmax(1) == labels).type(torch.float).sum().item()

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


        totalValLoss+=loss
        valCorrect+=(outputs.argmax(1)==labels).type(torch.float).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

trainCorrect = trainCorrect/len(train_loader.dataset)
valCorrect = valCorrect/len(val_loader.dataset)

avgTrainLoss = totalTrainLoss/trainSteps
avgValLoss = totalValLoss/valSteps

history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
history["train_acc"].append(trainCorrect)
history["val_loss"].append(avgValLoss.cpu().detach().numpy())
history["val_acc"].append(valCorrect)

print(classification_report(test_dataset))

end = time.time()
print("Time taken to train", np.round(end - start))

#Now, use test dataset:

with torch.no_grad():
    model.eval()

    preds = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs=model(inputs)
        preds.extend(outputs.argmax(axis=1).cpu().numpy())


print(classification_report(test_dataset.targets.cpu().numpy(), np.array(preds, target_names=test_dataset.classes)))


import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="trainloss")
plt.plot(history["val_loss"], label="val_loss")
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss and Accuracy")
plt.legend()


torch.save(model, args["model"])