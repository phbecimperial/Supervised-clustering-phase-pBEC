import pickle as pkl
import numpy as np
from os.path import sep
import cv2
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import select_files
import scienceplots
from generate_training import modelist
import power_length_analysis


if __name__ == '__main__':

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [7.2, 7.2],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

#     modelist = [
#     ([0,0], False, 0), ([0,1], False, 155 - 90), 
#     ([0,4], False, 70 + 90), ([0,6], False, 70 + 90),  ([0,9], False, 70 + 90),
# ]
    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pkl.load(f)


    stim_files, stim_mask = select_files.select_stimulated(files, 50)

    data = power_length_analysis.data_dict(files)


    data['PCA_Length'] = power_length_analysis.fit_pca(data['Lengths'], data['Pcas'])


    spect_map = matplotlib.cm.get_cmap('brg')

    # label_files = glob('Apr_9_predicted_labels_*.pkl')

    with open('Apr_2401_CNN_out.pkl', 'rb') as f:
        outs, preds = pkl.load(f)

    label_list = np.unique(preds, axis=0)


    for i, label in enumerate(label_list):
        mask = np.array([np.array_equal(p, label) for p in preds])
        color = spect_map((i+1)/(len(label_list)+1))

        plt.scatter(data['Lengths'][stim_mask][mask],
        data['Powers'][stim_mask][mask], color = color, zorder = 100, label = None)

    
    # plt.show()
    fig = plt.figure(figsize=[6.3, 4])
    fig, axes, gs = power_length_analysis.grid_plot(outs.shape[1],outs.shape[1]//2,2, 0.3, fig = fig)


    vs = [np.min(preds), np.max(preds)]


    fig.supxlabel('$\lambda$ ($nm$)')
    fig.supylabel('Pump power $(W)$')

    for i in range(outs.shape[1]):
        
        # color = spect_map((i+1)/(outs.shape[1]+1))

        # color = 'black'

        mode_prob = outs[:,i,0]

        # mode_prob = np.where(mode_prob > 0.5, mode_prob, np.nan)
        # mode_prob = mode_prob ** 2

        _,_, plot = power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
                                                data['Powers'][stim_mask],
                                                mode_prob, [940,960],
                                                [min(data['Powers']), max(data['Powers'])], cmap='Spectral_r',
                                                fig = fig, ax = axes[i], vs = vs)
        
        axes[i].set_title(modelist[i][0])
    
    cbax = plt.subplot(gs[1])
    plt.colorbar(mappable=plot,cax=cbax, cmap = 'Spectral_r', label = 'Likelihood') #, boundaries = np.linspace(0,1,100,endpoint=False))

    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\CNN_likelihood.png')
    plt.show()

