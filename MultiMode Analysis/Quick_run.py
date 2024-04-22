import generate_training
import Multi_loop
import time
from LightPipes import * 
import torch
import pickle as pkl
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from mode_classifier import ResNet, ResidualBlock
from tqdm import tqdm as tqdm
from pickle_Dataset import pickle_Dataset
import gc
from torch.cuda.amp import autocast, GradScaler

if __name__ == '__main__':
    t = time.localtime()
    save = True
    save_dir = r'C:\Users\Pouis\Documents\Uni Shit\Masters\Training Images'
    num_threads = 14
    ims = generate_training.generate_data_multithreaded(num_threads, 35000 // num_threads, 2500*um, 400, generate_training.modelist, [100*um, 220*um], fringe_size=[0.5, 0.8], save=save, mult_las_split=0, save_dir=save_dir)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    classes = len(generate_training.modelist)
    epochs = 40
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.00015
    val_split = 0.2
    batch_size = 128
    best_accuracy = 0.0

 
    transform = v2.Compose([v2.ToTensor(), v2.Resize((224,224), antialias=True), v2.Normalize((0.5,), (
        0.5,))])

    train_dataset = pickle_Dataset(root = r'C:\Users\Pouis\Documents\Uni Shit\Masters\Training Images', transforms = transform)


    numTrainSamp = round(len(train_dataset) * (1 - val_split))
    numValSamp = len(train_dataset) - numTrainSamp



    (train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)],
                                                                                    generator=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True, num_workers=2)




    for i in tqdm(range(classes), leave=True):
        model = ResNet(ResidualBlock, [2, 2, 2, 2], 2)

        # append_dropout(model)
        history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_loss": []
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0)
        model, history = Multi_loop.Training(model, epochs, i, optimizer, train_loader, val_loader, history, criterion=criterion)

        with open(r'MultiMode Analysis\Models\Apr22_Res_Class_' + str(i) + 'history', 'wb') as f:
            pkl.dump(history, f)

        del model

