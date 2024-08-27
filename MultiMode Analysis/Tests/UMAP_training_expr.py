import UMAP
import umap
import numpy as np
import power_length_analysis
from glob import glob as glob 
import matplotlib.pyplot as plt
import select_files
import Meta_classifier
import random
import pickle as pkl


if __name__ == "__main__":

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [6.3, 6.3],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })


    training_files = glob(r'C:\Users\Pouis\Documents\Uni Shit\Masters\Training Images\*.pkl')
    
    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        exp_files = pkl.load(f)

    data = power_length_analysis.data_dict(exp_files)
    stim_files, stim_mask = select_files.select_stimulated(data['Files'], 50)
    
    sigm_eimg = Meta_classifier.sigmoid(data['Images'][stim_mask], 0.05, 0.1)

    sigm_eimg = sigm_eimg[300:]

    training_files = random.sample(training_files, 300)
    
    training_images = []
    for i, file in enumerate(training_files):
        with open(file, 'rb') as f:
            timg, label = pkl.load(f)
        training_images.append(timg[2:-2,2:-2].flatten())
    
    all_imgs = np.concatenate([np.array(training_images), sigm_eimg])

    all_embed = UMAP.umap2d_V2(np.array(all_imgs), 3, 0.9, 2)

    plt.scatter(all_embed[:,0][:300],all_embed[:,1][:300])
    plt.scatter(all_embed[:,0][300:],all_embed[:,1][300:])
    plt.show()



