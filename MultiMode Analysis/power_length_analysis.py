import os
from os.path import sep
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from scipy.ndimage import center_of_mass
import Meta_classifier
import matplotlib
import select_files


def crop_save_image(files,size,root):

    for f in glob(root + r'\*'):
        os.remove(f)

    for f in files:
        image = cv2.imread(f, 0)
        name = f.split(sep)[-1][:-4]
        
        cx, cy  = center_of_mass(image**2)
        
        image_crop = image[int(int(cx) - size[0]/2):int(int(cx) + size[0]/2),
            int(int(cy) - size[1]/2):int(int(cy) + size[1]/2)]
        print(int(int(cx) - size[0]/2),int(int(cx) + size[0]/2))
        print(int(int(cy) - size[0]/2),int(int(cy) + size[0]/2))
        print(cx,cy)

        flag =  cv2.imwrite(root + r"\\" + 'Crop' + name + '.png', image_crop)
        print(root + r'\\' + name + '.png')
    ##return image_crop


def bec_crop_centre(bec_file: str, files: list[str], size: int, root: str):
    """
    New image cropping fn, parse in file of bec image and will take as center for all other images.
    V simple don't know why I didn't think of this before
    """
    bec_im = cv2.imread(bec_file)

    cx, cy = center_of_mass(bec_im**3)

    for f in glob(root + r'\*'):
        os.remove(f)

    for f in files:
        image = cv2.imread(f, 0)
        name = f.split(sep)[-1][:-4]
        
        image_crop = image[int(int(cx) - size[0]/2):int(int(cx) + size[0]/2),
            int(int(cy) - size[1]/2):int(int(cy) + size[1]/2)]
        #print(int(int(cx) - size[0]/2),int(int(cx) + size[0]/2))
        #print(int(int(cy) - size[0]/2),int(int(cy) + size[0]/2))
        #print(cx,cy)

        flag =  cv2.imwrite(root + r"\\" + 'Crop' + name + '.png', image_crop)
        #print(root + r'\\' + name + '.png')



def fit_pca(lengths, pcas):
    # plt.scatter(pcas, lengths)
    # plt.show()

    popt = np.polyfit(pcas, lengths, 1)
    fit = np.poly1d(popt)

    return fit(pcas)



def plot_2dhist(data_x, data_y, x_range, y_range, color):
    density, _, _ = np.histogram2d(data_x, 
                                   data_y,
                                   bins = 12, density=True, 
                                   range=[x_range,y_range])
    
    density = density/np.max(density)

    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap'+str(i),[color,color],256)
    
    cust_cmap._init()
    
    alphas = np.linspace(0, 1, cust_cmap.N+3)
    alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * 0.4
    cust_cmap._lut[:,-1] = alphas

    # plt.imshow(density.T, 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap, origin='lower')

    plt.imshow(density.T, interpolation='bicubic',
               interpolation_stage='rgba', origin='lower', 
               extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               aspect='auto', cmap=cust_cmap)



if __name__ == '__main__':

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
        #plt.imshow(image
        #plt.show()
        #crop_save_image(image, (180,180), 'pbecCrop' + split_file1[1][3:-4], r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\MultiMode Analysis\20240222')
        #plt.imshow(image)
        #'plt.show()
        split_file = split_file1[-1].split('_')
        print(split_file)
        powers.append(float(split_file[4]))
        lengths.append(float(split_file[5]))
        int_times.append(float(split_file[3]))
        pcas.append(float(split_file[-2]))
        #images.append(image)


    data = {
        'Files': np.array(files),
        'Powers': np.array(powers),
        'Lengths': np.array(lengths),
        'Int_times': np.array(int_times),
        'Pcas': np.array(pcas)
    }


    data['PCA_Length'] = fit_pca(data['Lengths'], data['Pcas'])


    spect_map = matplotlib.cm.get_cmap('brg')







    # phases = []
    # for i in np.unique(cluster_labels):
    #     cluster_idx = np.argwhere(cluster_labels == i)

    #     fig, axes = plt.subplots(nrows=2, ncols=3)


    #     count = 0 
    #     while count < 6:
    #         row = count // 3
    #         col = count % 3
    #         idx = np.random.randint(len(cluster_idx))
    #         image = cv2.imread(data['Files'][stim_mask][cluster_idx[idx]][0], 0)
    #         axes[row,col].imshow(image)
    #         count += 1

    #     plt.show()
    #     in_phase = input("Enter Cluster Label: ")
    #     phases.append(in_phase)
        

    # plot_2dhist(data['PCA_Length'][np.invert(stim_mask)], data['Powers'][np.invert(stim_mask)],
    #             [min(data['PCA_Length']), max(data['PCA_Length'])],
    #             [min(data['Powers']), max(data['Powers'])],
    #             'grey')

    # plt.scatter(data['PCA_Length'][np.invert(stim_mask)],
    #             data['Powers'][np.invert(stim_mask)], color = 'grey', zorder = 100, label = 'Thermal Cloud')

    label_files = glob('Mar_20_21_predicted_labels_*.pkl')

    # for file in label_files:

    # with open('Mar_20_21_NoPow_predicted_labels_8.pkl', 'rb') as f:
    #     cluster_labels = pkl.load(f)

    with open('Mar_20_21_CNN_out.pkl', 'rb') as f:
        outs, preds = pkl.load(f)

    
    cluster_labels = preds

    for i in np.unique(cluster_labels):
        mask = cluster_labels  == i
        color = spect_map((i+1)/(max(np.unique(cluster_labels)+1)))

        plot_2dhist(data['Lengths'][stim_mask][mask],
                    data['Powers'][stim_mask][mask],
                    [min(data['Lengths']), max(data['Lengths'])],
                    [min(data['Powers']), max(data['Powers'])],
                    color
                    )
        
        plt.scatter(data['Lengths'][stim_mask][mask],
                    data['Powers'][stim_mask][mask], color = color, zorder = 100, label = None)


    l_points = [[945, 0.18], [950,0.12]]
    h = 0.015

    line_files, line, line_params, line_lengths = select_files.select_line(stim_files, 
                                                                        l_points, h)


    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.grid(c='black')
    # clusters = file.split('_')[-1].split('.')[0]
    # plt.title(f'Num Clusters: {clusters}')
    ax = plt.gca()

    ax = select_files.plot_line(ax, line, line_params, h, l_points)

    for i in ax.spines:
        ax.spines[i].set_color('w') 
    ax.tick_params(color = 'w')
    ax.yaxis.label.set_color('w')
    ax.xaxis.label.set_color('w')
    plt.ylabel('Pump Power (W)')
    plt.xlabel('Cavity Length (nm)')

    plt.tight_layout()

    plt.show()


    fig, axes = select_files.show_line_images(data['Files'][stim_mask],
                                line_files, 8,
                                cluster_labels, log = True)

    plt.show()

    plt.scatter(data['Lengths'][np.invert(stim_mask)], data['Powers'][np.invert(stim_mask)], color = 'grey')
    plt.scatter(data['Lengths'][stim_mask], data['Powers'][stim_mask], c =cluster_labels, cmap='tab10')
    plt.show()