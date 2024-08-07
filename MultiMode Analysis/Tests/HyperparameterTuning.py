import torch
from pickle_Dataset import pickle_Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from mode_classifier import ResNet, ResidualBlock
from tqdm import tqdm as tqdm
from Multi_loop import Training
import optuna
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def objective(trial):

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_uniform('weight decay', 1e-5, 0.5)
    momentum = trial.suggest_uniform('Momentum', 0.5, 0.99)
    layer0 = trial.suggest_categorical('layer0', [1,2,3])
    layer1 = trial.suggest_categorical('layer1', [1,2,3])
    layer2 = trial.suggest_categorical('layer2', [1,2,3])
    layer3 = trial.suggest_categorical('layer3', [1,2,3])




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = 10
    epochs = 50
    val_split = 0.2
    batch_size = 64
    label = 4

    transform = v2.Compose([v2.ToTensor(), v2.Resize((224, 224)), v2.Normalize((0.5,), (
        0.5,))])

    train_dataset = pickle_Dataset(root=r'Training_Images_2', transforms=transform)

    numTrainSamp = round(len(train_dataset) * (1 - val_split))
    numValSamp = len(train_dataset) - numTrainSamp

    criterion = torch.nn.BCEWithLogitsLoss()


    (train_dataset, validate_dataset) = torch.utils.data.random_split(train_dataset, [int(numTrainSamp), int(numValSamp)],
                                                                      generator=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ResNet(ResidualBlock, [layer0, layer1, layer2, layer3], 2)

    # append_dropout(model)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_loss": []
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    model, history = Training(model, epochs, label, optimizer, train_loader, val_loader, history, criterion=criterion)

    print(max(history['val_accuracy']))
    return max(history['val_accuracy'])

if __name__ == "__main__":
    # Create an Optuna study object and optimize
    study = optuna.create_study(direction='maximize', study_name='[0, 4] optimisation')
    study.optimize(objective, n_trials=40)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)