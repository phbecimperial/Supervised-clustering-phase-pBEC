import torch
import pickle as pkl
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from mode_classifier import ResNet, ResidualBlock
from tqdm import tqdm as tqdm
from pickle_Dataset import pickle_Dataset
import gc
from torch.cuda.amp import autocast, GradScaler
#import generate_training



def Training(model, epochs, label, optimizer, train_loader, val_loader, history, criterion, stop_check = 15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    iter = tqdm(range(epochs), leave=True)
    strikes = 0
    scaler = GradScaler()

    for i in iter:
        model.train()

        steps = 0
        correct = 0
        total = 0
        total_loss = 0
        for inputs, keys in train_loader:

            inputs, keys = inputs.to(device), torch.select(keys, 1, label).to(device)
            with autocast():
                #inputs = inputs.to(memory_format=torch.channels_last)
                #model=model.to(memory_format=torch.channels_last)
                outputs = model(inputs)
                optimizer.zero_grad()

                keys = keys.float()
                loss = criterion(outputs, keys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            total_loss += loss
            #optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += keys.size(0)
            _, correct_keys =  torch.max(keys.data, 1)
            correct += (predicted == correct_keys).sum().item()

            steps += 1
            del inputs, keys, outputs
            torch.cuda.empty_cache()
            gc.collect()

        
        history['train_accuracy'].append(correct / total)
        history['train_loss'].append(total_loss.cpu().detach().numpy()/ steps)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            steps = 0
            for images, keys in val_loader:
                images = images.to(device)
                keys = torch.select(keys, 1, label).to(device)
                #images = images.to(memory_format=torch.channels_last)
                #model=model.to(memory_format=torch.channels_last)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += keys.size(0)
                _, correct_keys =  torch.max(keys.data, 1)
                correct += (predicted == correct_keys).sum().item()

                outputs = torch.sigmoid(outputs)
                keys = keys.float()
                loss = criterion(outputs, keys)
                total_loss += loss
                steps += 1
                del images, keys, outputs


            iter.set_description(f'Accuracy of the network on the validation images: {100 * correct / total} %')
            if (correct/total) > max(history['val_accuracy'], default=0):
                torch.save(model, r'Models\Scuffed_Res_Class_' + str(label) + '.pt', pkl)
                # del model


            history['val_accuracy'].append(correct / total)
            history['val_loss'].append(loss.cpu().detach().numpy() / steps)


        

        # if i > stop_check:
        #
        #     train_window = np.array(history['train_accuracy'])[i-(stop_check - 1):i]
        #     train_fit = np.polyfit(np.arange(train_window.size), train_window, deg=1)
        #
        #     val_window = np.array(history['val_accuracy'])[i-(stop_check - 1):i]
        #     val_fit = np.polyfit(np.arange(val_window.size), val_window, deg=1)
        #
        #
        #     if train_fit[0] > 0 and val_fit[0] < 0:
        #         strikes += 1
        #     else:
        #         strikes = 0
        #     if strikes > 3:
        #         return model, history

    return model, history

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = 6
    epochs = 30
    criterion = torch.nn.BCEWithLogitsLoss()
    learning_rate = 0.01
    val_split = 0.2
    batch_size = 128
    best_accuracy = 0.0


    transform = v2.Compose([v2.ToTensor(), v2.Resize((224,224), antialias=True), v2.Normalize((0.5,), (
        0.5,))])

    train_dataset = pickle_Dataset(root = r'Training_Images', transforms = transform)


    numTrainSamp = round(len(train_dataset) * (1 - val_split))
    numValSamp = len(train_dataset) - numTrainSamp



    (train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)],
                                                                                    generator=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True, num_workers=2)




    for i in tqdm(range(classes), leave=True):
        model = ResNet(ResidualBlock, [2, 1, 1, 1], 2)

        # append_dropout(model)
        history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_loss": []
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.01, momentum = 0.9)
        model, history = Training(model, epochs, i, optimizer, train_loader, val_loader, history, criterion=criterion)

        with open(r'Models\Scuffed_Res_Class_' + str(i) + 'history', 'wb') as f:
            pkl.dump(history, f)

        del model




