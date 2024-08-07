import cv2
from scipy.ndimage import zoom
import glob
import numpy as np
from scipy.ndimage import center_of_mass
from os.path import sep
import cv2
from os.path import sep
import umap
import UMAP as cu
import matplotlib.pyplot as plt
import scienceplots
from sklearn.cluster import KMeans
import tqdm
import gc
from generate_training import modelist
import matplotlib
#import hdbscan
# import UMAP_utils as uu
import pickle as pkl
import power_length_analysis
import select_files
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
import Meta_classifier


if __name__ == "__main__":
    
    plt.rcParams.update({
        'figure.figsize': [6.3, 6.3],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    # Replace with your file list
    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pkl.load(f)


    data = power_length_analysis.data_dict(files)
    stim_files, stim_mask = select_files.select_stimulated(data['Files'], 50)

    with open('Apr_26_features.pkl', 'rb') as f:
        cnn_features = pkl.load(f)

    #uncomment for image embeddings
    # img_embed = cu.umap2d_V2(data['Images'][stim_mask], 4, 0, 2)

    #uncomment for cnn embedings
    cnn_embed = cu.umap2d_V2(cnn_features, 16, 0.0, 2)

    num_clusters = 5
    # cnn_embed / cnn_features to change clusering
    membership_mat = Meta_classifier.quick_fcmeans(cnn_features, num_clusters = num_clusters, m = 2, return_matrix= True)

    # dont change for nice plots
    fig = plt.figure(figsize=[6.3, 4])
    fig, axes, gs = power_length_analysis.grid_plot(num_clusters,num_clusters//2, 2, 0.3, fig = fig)

    # Sets range for colors in imshow
    vs = [np.min(membership_mat), np.max(membership_mat)]
    for i, col in enumerate(membership_mat):

        #uncomment for umap plot
        plot = axes[i].scatter(cnn_embed[:,0], cnn_embed[:,1], c = col, cmap = 'Spectral_r',s = 0.8, vmin = vs[0], vmax = vs[1])

        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])

        axes[i].set_title(f'Cluster {i + 1}')

        #uncomment for PL plot
        # _,_, plot = power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
        #     data['Powers'][stim_mask],
        #     col,
        #     [940, 960],
        #     [min(data['Powers']), max(data['Powers'])],
        #     cmap = 'Spectral_r', fig=fig, ax = axes[i], vs = vs
        #     )
    
    #colorbar stuff
    cbax = plt.subplot(gs[1])
    plt.colorbar(plot, cbax,cmap = 'Spectral_r', label = 'Likelihood',)
    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\UMAP plots\Umap_cnn_Cluster_feat.pdf', format = 'pdf')
    plt.show()