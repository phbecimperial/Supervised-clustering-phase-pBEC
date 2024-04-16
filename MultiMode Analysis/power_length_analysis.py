import os
from os.path import sep
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from scipy.ndimage import center_of_mass
from scipy.stats import binned_statistic_2d
import Meta_classifier
import matplotlib
import select_files
import matplotlib.gridspec as gridspec 
import matplotlib.ticker as ticker 


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
    bec_im = cv2.imread(bec_file, 0)

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


def grid_plot(nplots ,ncols, nrows, wspace, tick_spacing = 10):
    fig = plt.figure(figsize=[7.2, 3.6])
    axes = []
    if nplots % 2 == 0:
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=wspace)

        for i in range(ncols*2):
            axes.append(fig.add_subplot(gs[i//ncols,i%ncols], title = i))
            if i % ncols != 0:
                axes[i].set_yticklabels([])
            if i//ncols == 0:
                axes[i].set_xticklabels([])

    else:
        gs = gridspec.GridSpec(nrows=nplots, ncols=1)
        gs01 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols=ncols, subplot_spec=gs[1])
        gs02 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols=(nplots - ncols), 
                                                subplot_spec=gs[0], wspace=wspace)
        for i in range(ncols):
            axes.append(fig.add_subplot(gs01[i]))
            if i != 0:
                axes[i].set_yticklabels([])
        for i in range(nplots - ncols):
            axes.append(fig.add_subplot(gs02[i]))
            if i != 0:
                axes[i + ncols].set_yticklabels([])
        
    for i, _ in enumerate(axes):
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    return fig, axes, gs

def plot_2d_stat_hist(data_x, data_y, alphas, x_range, y_range, color, fig = None, ax = None):

    # density, xedges, yedges = np.histogram2d(data_x, 
    #                                data_y,
    #                                bins = 12, density=True, 
    #                                range=[x_range,y_range])
    
    
    statistic, _,_,_ = binned_statistic_2d(data_x, data_y, 
                                         alphas, bins=30, 
                                         range = [x_range,y_range])

    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',['white',color],256)
    
    cust_cmap._init()
    
    statistic = np.nan_to_num(statistic)

    alphas = np.linspace(0, 1, cust_cmap.N+3)
    # alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * 0.4
    # cust_cmap._lut[:,-1] = alphas


    if fig is None:
        fig, ax = plt.subplots()



    ax.imshow(statistic.T, 
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            aspect='auto', cmap=cust_cmap, origin='lower')

    # plt.imshow(statistic.T, interpolation='bicubic',
    #            interpolation_stage='rgba', origin='lower', 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap)

    return fig, ax   

def plot_2dhist(data_x, data_y, x_range, y_range, color):
    density, _, _ = np.histogram2d(data_x, 
                                   data_y,
                                   bins = 12, density=True, 
                                   range=[x_range,y_range])
    
    density = density/np.max(density)

    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',[color,color],256)
    
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


    data['PCA_Length'] = fit_pca(data['Lengths'], data['Pcas'])


    spect_map = matplotlib.cm.get_cmap('brg')

    label_files = glob('Apr_16_POWLEN_predicted_labels_*.pkl')

    for file in label_files:

        # with open('Apr_2_NoPow_predicted_labels_9.pkl', 'rb') as f:
        #     cluster_labels = pkl.load(f)

        # with open('Apr_2_CNN_out.pkl', 'rb') as f:
        #     outs, preds = pkl.load(f)

        with open(file, 'rb') as f:
            cluster_labels = pkl.load(f)
        
        # cluster_labels = preds


        for i in np.unique(cluster_labels, axis=0):
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


        l_points = [[950, 0.15], [960,0.2]]
        h = 0.015

        line_files, line, line_params, line_lengths = select_files.select_line(stim_files, 
                                                                            l_points, h)


        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.grid(c='black')
        # clusters = file.split('_')[-1].split('.')[0]

        plt.title(f'Num Clusters: {len(np.unique(cluster_labels))}')
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
                                    cluster_labels, log = False, cmap='brg')

        plt.show()

        plt.scatter(data['Lengths'][np.invert(stim_mask)], data['Powers'][np.invert(stim_mask)], color = 'grey')
        plt.scatter(data['Lengths'][stim_mask], data['Powers'][stim_mask], c =cluster_labels, cmap='tab10')
        plt.show()