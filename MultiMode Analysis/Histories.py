import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle as pkl


files = glob(r'Models\\*history*')

histories = []
for i in files:
    with open(i, 'rb') as f:
        hist = pkl.load(f)
        new_hist = {
            "train_loss": [],
            "train_accuracy": hist['train_accuracy'],
            "val_accuracy": hist["val_accuracy"]
            }
        
        for _, loss in enumerate(hist['train_loss']):
            if type(loss) != np.float64:
                new_hist['train_loss'].append(loss.detach().cpu().numpy())
            else:
                new_hist['train_loss'].append(loss)

        histories.append(new_hist)


for i, hist in enumerate(histories):
    #i = 11
    plt.plot(histories[i]['train_accuracy'], label = 'Train Accuracy')
    plt.plot(histories[i]['val_accuracy'], label = 'Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(files[i])
    plt.legend()
    plt.show()


