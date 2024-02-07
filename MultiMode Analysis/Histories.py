import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle as pkl


files = glob(r'MultiMode Analysis\\Models\\*history*')

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
            new_hist['train_loss'].append(loss.detach().cpu().numpy())

        histories.append(new_hist)


for i, hist in enumerate(histories):
    plt.plot(hist['train_loss'], label = 'model' + str(i) + 'loss')
#    plt.plot(hist['val_accuracy'], label = 'model' + str(i) + 'loss')
    plt.show()


