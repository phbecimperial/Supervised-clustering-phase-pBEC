from os.path import sep
from glob import glob
import pickle
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib
import matplotlib.pyplot as plt
import cv2
import power_length_analysis
import select_files
import Meta_classifier
import scienceplots
import matplotlib.gridspec as gridSpec
import matplotlib.ticker as ticker 



if __name__ == '__main__':

    plt.style.use(['science', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [7.2, 7.2],
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



    powers = []
    lengths = []
    int_times = []
    images = []
    pcas = []
    for i, file in enumerate(files):
        
        split_file1 = file.split(sep)
        #print(split_file1)
        image = cv2.imread(file)


        split_file = split_file1[-1].split('_')
        #print(split_file)
        powers.append(float(split_file[4]))
        lengths.append(float(split_file[5]))
        int_times.append(float(split_file[3]))
        pcas.append(float(split_file[-2]))


    data = {
        'Files': np.array(files),
        'Powers': np.array(powers),
        'Lengths': np.array(lengths),
        'Int_times': np.array(int_times),
        'Pcas': np.array(pcas)
    }


    data['PCA_Length'] = power_length_analysis.fit_pca(data['Lengths'], data['Pcas'])



    with open('Apr_20_POWLEN_features.pkl', 'rb') as f:
        features = pickle.load(f)   

    labels, alphas = Meta_classifier.quick_fcmeans(features, num_clusters= 6, m= 1)

    spect_map = matplotlib.cm.get_cmap('brg')

    fig, ax = plt.subplots()

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label

        color = spect_map((i+1)/(max(np.unique(labels)+1)))

        power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask][mask],
                    data['Powers'][stim_mask][mask],
                    alphas[mask],
                    [940, max(data['Lengths'])],
                    [min(data['Powers']), max(data['Powers'])],
                    color, fig, ax
                    )
    
    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\fc_CNN_overlay_7_clusters.png')

    plt.show()

    print([min(data['Lengths']), max(data['Lengths'])])

    membership_mat = Meta_classifier.quick_fcmeans(features, num_clusters = 6, m = 1, return_matrix= True)


    numrows = 2
    numcols = membership_mat.shape[0]//numrows
    
    fig = plt.figure(figsize=[7.2, 3.6])
    axes = []
    if membership_mat.shape[0] % 2 == 0:
        gs = gridSpec.GridSpec(nrows=numrows, ncols=numcols)

        for i in range(numcols*2):
            axes.append(fig.add_subplot(gs[i//numcols,i%numcols], title = i))
            if i % numcols != 0:
                axes[i].set_yticklabels([])

    else:
        gs = gridSpec.GridSpec(nrows=numrows, ncols=1)
        gs01 = gridSpec.GridSpecFromSubplotSpec(nrows = 1, ncols=numcols, subplot_spec=gs[1])
        gs02 = gridSpec.GridSpecFromSubplotSpec(nrows = 1, ncols=(membership_mat.shape[0] - numcols), 
                                                subplot_spec=gs[0], wspace=0.3)
        for i in range(numcols):
            axes.append(fig.add_subplot(gs01[i]))
            if i != 0:
                axes[i].set_yticklabels([])
        for i in range(membership_mat.shape[0] - numcols):
            axes.append(fig.add_subplot(gs02[i]))
            if i != 0:
                axes[i + numcols].set_yticklabels([])
        
    for i, _ in enumerate(axes):
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(10))

    

    for i, col in enumerate(membership_mat):
        color = spect_map((i+1)/(membership_mat.shape[0]+1))





        power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
            data['Powers'][stim_mask],
            col,
            [940, 960],
            [min(data['Powers']), max(data['Powers'])],
            color, fig=fig, ax = axes[i]
            )
    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\Ind_clusters.png')
    # plt.show()



