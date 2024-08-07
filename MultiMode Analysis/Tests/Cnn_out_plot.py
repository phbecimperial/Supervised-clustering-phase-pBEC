import pickle as pkl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import select_files
from scipy.interpolate import interp1d
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

    with open('Apr_26_CNN_out.pkl', 'rb') as f:
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

    stats = []
    save_dict = {}
    for i in range(outs.shape[1]):
        
        # color = spect_map((i+1)/(outs.shape[1]+1))

        # color = 'black'

        mode_prob = outs[:,i,0]

        # mode_prob = np.where(mode_prob > 0.5, mode_prob, np.nan)
        # mode_prob = mode_prob ** 2

        _,_, plot, stat = power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
                                                data['Powers'][stim_mask],
                                                mode_prob, [940,960],
                                                [min(data['Powers']), max(data['Powers'])], cmap='Spectral_r',
                                                fig = fig, ax = axes[i], vs = vs, ret_stat=True)
        stats.append(stat)
        axes[i].set_title(modelist[i][0])

        save_dict[f'mode {modelist[i][0]} probabilities'] = stat[0].tolist()
    
    save_dict['x edges'] = stat[1].tolist()
    save_dict['y edges'] = stat[2].tolist()
    
    cbax = plt.subplot(gs[1])
    plt.colorbar(mappable=plot,cax=cbax, cmap = 'Spectral_r', label = 'Likelihood') #, boundaries = np.linspace(0,1,100,endpoint=False))

    # plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\CNN plots\CNN_likelihood.pdf', format='pdf')
    plt.show()

    plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['k', 'r', 'b', 'g', 'm']) + cycler('linestyle', ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))])")

    import json

    with open('Cnn_probabilities.json', 'w') as f:
        json.dump(save_dict, f)
        

    for i, stat in enumerate(stats):
        idx = 12
        probs = stat[0].T[::-1]
        # plt.imshow(probs)
        # plt.show()
        n_mask = np.isnan(probs[:,idx])
        yval = stat[1][-idx]
        

        xs = stat[2][1:-1][~n_mask]
        print(len(xs))
        x = np.linspace(min(xs),max(xs),100)

        
        spline = interp1d(xs, probs[:,idx][~n_mask])
        plt.plot(x, spline(x), label = modelist[i][0])
        plt.title(yval)
    plt.legend()
    plt.show()
    


