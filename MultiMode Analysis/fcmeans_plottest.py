from os.path import sep
from glob import glob
import pickle
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib
import UMAP as cu
import matplotlib.pyplot as plt
import cv2
import power_length_analysis
import select_files
import Meta_classifier
import scienceplots
import matplotlib.gridspec as gridSpec
import matplotlib.ticker as ticker 



if __name__ == '__main__':

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [6.3, 6.3],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pickle.load(f)

    # bec_crop_centre(r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\20240321\pbec_20240321_000118_7814.0_0.12576051724137932_941.7710571289062_7.275862068965518_.png",
    #                 files, (220,220), 
    #                 root = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321")


    stim_files, stim_mask = select_files.select_stimulated(files, 50)



    data = power_length_analysis.data_dict(files)

    with open('Apr_26_features.pkl', 'rb') as f:
        features = pickle.load(f)   


    data['PCA_Length'] = power_length_analysis.fit_pca(data['Lengths'], data['Pcas'])

    num_clusters = 7




    labels, alphas = Meta_classifier.quick_fcmeans(features, num_clusters= num_clusters, m= 3)

    #pl plot
    fig, ax, _, _ = power_length_analysis.all_cluster_plot(
        np.max(labels), labels,
        data['Lengths'][stim_mask], data['Powers'][stim_mask],
        'tab20b', 30, [940,960], [min(data['Powers']), max(data['Powers'])], 8
        )
    fig.set_figwidth(4.72)
    fig.set_figheight(4.72)
    power_length_analysis.add_loss_rate(ax, 'inter.pkl')
    ax.set_xlabel('$\lambda$ ($nm$)')
    ax.set_ylabel('Pump Power (W)')
    # plt.savefig(rf'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\FCplots\FC_Fake_Kmeans_{num_clusters}.pdf', format = 'pdf')
    plt.show()
    

    #umap plot
    # labels , _ = Meta_classifier.Kmeans_no_CNN(data['Files'][stim_mask], 7)
    # img_embed = cu.umap2d_V2(data['Images'][stim_mask], 16, 0.2, 2)
    # # cnn_embed = cu.umap2d_V2(features, 16, 0.0, 2)

    # fig, ax = plt.subplots(figsize = [3.7, 2.5])
    # mapp = ax.scatter(img_embed[:,0], img_embed[:,1], c = (labels + 0.5)/(num_clusters + 0.5), cmap = 'tab20b',s = 5, vmin = 0, vmax = 1)

    # ax.set_yticklabels([])
    # ax.set_xticklabels([])

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=num_clusters + 0.5)
    # cbarmap = matplotlib.colormaps['tab20b']
    # my_cmap = cbarmap(np.arange(cbarmap.N))
    # my_cmap[:,-1] = np.ones_like(my_cmap[:,-1])

    # cbarmap = matplotlib.colors.ListedColormap(my_cmap)

    # mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cbarmap)

    # cbar = fig.colorbar(mappable, ax=ax, boundaries = np.arange(0, stop = num_clusters + 0.5))
    # tick_locs = (np.arange(0, num_clusters)) + 0.5
    # cbar.set_ticks(tick_locs)
    # cbar.set_ticklabels(np.arange(num_clusters))

    # plt.savefig(rf'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\UMAP plots\UMAP_img_Just_Kmeans{num_clusters}.pdf', format = 'pdf')
    # plt.show()
    

    # print([min(data['Lengths']), max(data['Lengths'])])

    #ind plot

    membership_mat = Meta_classifier.quick_fcmeans(features, num_clusters = num_clusters, m = 3, return_matrix= True)


    numrows = 2
    numcols = membership_mat.shape[0]//numrows
    
    fig = plt.figure(figsize=[6.3, 4])

    fig, axes, gs = power_length_analysis.grid_plot(num_clusters,numcols, numrows, 0.3, fig = fig)

    fig.supxlabel('$\lambda$ ($nm$)', y = 0.005)
    fig.supylabel('Pump power $(W)$')



    vs = [np.min(membership_mat), np.max(membership_mat)]

    for i, col in enumerate(membership_mat):
        # color = spect_map((i+1)/(membership_mat.shape[0]+1))





        _,_, plot = power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
            data['Powers'][stim_mask],
            col,
            [940, 960],
            [min(data['Powers']), max(data['Powers'])],
            cmap = 'Spectral_r', fig=fig, ax = axes[i], vs = vs
            )
        
        axes[i].set_title(i)

    cbax = plt.subplot(gs[1])
    plt.colorbar(plot, cbax,cmap = 'Spectral_r', label = 'Likelihood',)

    # plt.savefig(rf'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\FCplots\Ind_clusters{num_clusters}.pdf', format = 'pdf')
    plt.show()

