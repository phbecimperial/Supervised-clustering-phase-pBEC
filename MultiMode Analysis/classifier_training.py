import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from mode_classifier import ResNet, ResidualBlock
from sklearn.model_selection import train_test_split
from glob import glob
import mgzip
import random
import pickle
import gc

def data_loader(num_batches, batch_size, dataset_path, split):

    normalize = v2.Normalize(
    mean=[0.4914],
    std=[0.2023],
    )

    transform = v2.Compose([
            v2.Resize((224,224)),
            normalize,
    ])

    path_list = glob(dataset_path + "\*")
    path_list = random.sample(path_list,num_batches*batch_size)


    ims = []
    keys = []
    for path in path_list:
        with mgzip.open(path,'rb') as f:
            item = pickle.load(f)
        ims.append(transform(item[0]))
        keys.append(item[1])
    
    xt, xT, yt, yT = train_test_split(ims, keys, train_size=split, random_state=123)

    return np.split(xt,num_batches), xT, np.split(yt,num_batches), yT


num_classes = 10
num_epochs = 20
batch_size = 32
num_batches = 5
learning_rate = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Specified layers for the resnet34 architecture
layers = [3, 4, 6, 3]
model = ResNet(ResidualBlock, layers).to(device)

x_train_batches, x_test, y_train_batches, y_test = data_loader(5,batch_size,'Training_images', 0.33)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  


for epoch in range(num_epochs):
    for i, (images, keys) in enumerate(zip(x_train_batches, y_train_batches)):
        images = images.to(device)
        keys = keys.to(device)

        outputs = model(images)
        loss = criterion(outputs, keys)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, keys, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, keys in zip(x_test, y_test):
            images = images.to(device)
            keys = keys.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += keys.size(0)
            correct += (predicted == keys).sum().item()
            del images, keys, outputs
    
        print(f'Accuracy of the network on the validation images: {100 * correct / total} %')


torch.save(model, 'Saved_models/Model_1')