import torch
import pickle as pkl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from mode_classifier import ResNet, ResidualBlock
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
import time
import numpy as np
import argparse
from pickle_Dataset import pickle_Dataset
import gc


def Training(model, epochs, label, optimizer, train_loader, val_loader, history):  
    model.to(device)
    iter = tqdm(range(epochs))
    for i in iter:
        model.train()

        correct = 0
        total = 0
        for inputs, keys in train_loader:
            inputs, keys = inputs.to(device), torch.select(keys, 1, label).to(device)
            outputs = model(inputs)
            optimizer.zero_grad()

            loss = criterion(outputs, keys)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += keys.size(0)
            _, correct_keys =  torch.max(keys.data, 1)
            correct += (predicted == correct_keys).sum().item()


            del inputs, keys, outputs
            torch.cuda.empty_cache()
            gc.collect()

        
        history['train_accuracy'].append(correct / total)
        history['train_loss'].append(loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, keys in val_loader:
                images = images.to(device)
                keys = torch.select(keys, 1, label).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += keys.size(0)
                _, correct_keys =  torch.max(keys.data, 1)
                correct += (predicted == correct_keys).sum().item()
                
                del images, keys, outputs
            history['val_accuracy'].append(correct / total)
            iter.set_description(f'Accuracy of the network on the validation images: {100 * correct / total} %')

        # if i > 10:
        #     train_avg = np.mean(np.gradient(np.array(history['train_accuracy'])[i-9:i]))
        #     val_avg = np.mean(np.gradient(np.array(history['val_accuracy'])[i-9:i]))

        #     if train_avg > 0 and val_avg < 0:
        #         return model, history

    return model, history



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = 13
epochs = 100
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
train_split = 0.5
batch_size = 64

transform = v2.Compose([v2.ToTensor(), v2.Resize((224,224)), v2.Normalize((0.5,), (
    0.5,))])

train_dataset = pickle_Dataset(root = r'MultiMode Analysis/Training_Images', transforms = transform)
test_dataset = pickle_Dataset(root = r'MultiMode Analysis/Training_Images', transforms = transform)


numTrainSamp = int(len(train_dataset)) * train_split
numValSamp = int(len(train_dataset)) * (1 - train_split)

(train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)],
                                                                  generator=None)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)



for i in tqdm(range(11, classes)):
    model = ResNet(ResidualBlock, [3, 4, 6, 3], 2)
    history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    model, history = Training(model, epochs, i, optimizer, train_loader, val_loader, history)
    torch.save(model, r'MultiMode Analysis\\Models\Res_Class_' + str(i), pkl)
    with open(r'MultiMode Analysis\\Models\Res_Class_' + str(i) + 'history', 'wb') as f:
        pkl.dump(history, f)
    del model




