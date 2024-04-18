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

    plt.style.use(['science', 'no-latex'])

    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pkl.load(f)


    stim_files, stim_mask = select_files.select_stimulated(files, 50)

    # crop_save_image(files, (220,220), root = r'C:\Users\Pouis\OneDrive - Imperial College London\202403_link\Cropped_Images\20240321')


    powers = []
    lengths = []
    int_times = []
    images = []
    pcas = []
    for i, file in enumerate(files):
        
        split_file1 = file.split(sep)
        print(split_file1)
        image = cv2.imread(file)


        split_file = split_file1[-1].split('_')
        print(split_file)
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


    spect_map = matplotlib.cm.get_cmap('brg')

    # label_files = glob('Apr_9_predicted_labels_*.pkl')

    with open('Apr_16_CNN_out.pkl', 'rb') as f:
        outs, preds = pkl.load(f)

    label_list = np.unique(preds, axis=0)

    print(label_list)

    for i, label in enumerate(label_list):
            mask = np.array([np.array_equal(p, label) for p in preds])
            color = spect_map((i+1)/(len(label_list)+1))

            plt.scatter(data['Lengths'][stim_mask][mask],
            data['Powers'][stim_mask][mask], color = color, zorder = 100, label = None)

    
    # plt.show()

    fig, axes, gs = power_length_analysis.grid_plot(outs.shape[1],outs.shape[1]//2,2, 0.3)

    fig.supxlabel('$\lambda$ ($nm$)')
    fig.supylabel('Pump power $(W)$')

    for i in range(outs.shape[1]):
        
        color = spect_map((i+1)/(outs.shape[1]+1))

        mode_prob = outs[:,i,0]

        power_length_analysis.plot_2d_stat_hist(data['Lengths'][stim_mask],
                                                data['Powers'][stim_mask],
                                                mode_prob, [940,960],
                                                [min(data['Powers']), max(data['Powers'])], color,
                                                fig = fig, ax = axes[i])
        
        axes[i].set_title(modelist[i][0])
    plt.show()


