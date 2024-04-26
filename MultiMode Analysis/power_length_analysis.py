import os
from os.path import sep
from glob import glob
import matplotlib.colors
import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from scipy.ndimage import center_of_mass
from scipy.stats import binned_statistic_2d
import Meta_classifier
import matplotlib
import scienceplots
import select_files
import matplotlib.gridspec as gridspec 
import matplotlib.ticker as ticker
from scipy.spatial import KDTree
from scipy.interpolate import griddata, CloughTocher2DInterpolator, interp1d


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


def add_loss_rate(ax: plt.Axes, abs_path):
    """_summary_

    Args:
        ax (plt.Axes): axes to add thermalisation to
        abs_path (str): path with interpolated absorption rates
    """

    with open(abs_path, 'rb') as f:
        loss_rate = pkl.load(f)

    llim, rlim = ax.get_xlim()
    x = np.linspace(llim, rlim, 2)

    # thermal = absorb(x)/70
    xto_thermal = lambda y:  loss_rate(y*1e-9)/8e10



    thermalto_x = interp1d(xto_thermal(x), x, fill_value='extrapolate')
    ax.secondary_xaxis('top', functions=(xto_thermal, thermalto_x), xscale = 'log', xlabel = '$\gamma$')

    

def bec_crop_centre(bec_file: str, files: list[str], size: list[int,int], root: str):
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
    """_summary_

    Args:
        lengths (_type_): _description_
        pcas (_type_): _description_

    Returns:
        np.poly1d : _description_
    """
    
    # plt.scatter(pcas, lengths)
    # plt.show()
    popt = np.polyfit(pcas, lengths, 1)
    fit = np.poly1d(popt)

    return fit(pcas)


def grid_plot(nplots ,ncols, nrows, wspace, hspace = 0.4, tick_spacing = 10, fig = None):
    if fig is None:
        fig = plt.figure(figsize=[6.3, 4])
    axes = []

    gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.08, width_ratios=[0.96, 0.04])

    if nplots % nrows == 0:
        gsp = gridspec.GridSpecFromSubplotSpec(nrows = nrows, ncols=ncols, hspace=hspace, subplot_spec=gs[0])

        for i in range(nplots):
            axes.append(fig.add_subplot(gsp[i//ncols,i%ncols]))
            if i % ncols != 0:
                axes[i].set_yticklabels([])
            if i < nplots - ncols:
                axes[i].set_xticklabels([])

    else:
        gsp = gridspec.GridSpecFromSubplotSpec(nrows=nrows, ncols=1, hspace=hspace, subplot_spec=gs[0])
        gs01 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols=ncols, subplot_spec=gsp[1])
        gs02 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols=(nplots - ncols), 
                                                subplot_spec=gsp[0], wspace=wspace)
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

def plot_2d_stat_hist(data_x, data_y, alphas, x_range, y_range, 
                      color = None, cmap = None, fig = None, ax = None, to_alpha = False,
                      vs = [0,1], ret_stat = False):

    # density, xedges, yedges = np.histogram2d(data_x, 
    #                                data_y,
    #                                bins = 12, density=True, 
    #                                range=[x_range,y_range])
    
    
    statistic, x_edge, y_edge ,_ = binned_statistic_2d(data_x, data_y, 
                                         alphas, bins=30, 
                                         range = [x_range,y_range])

    
    if color is not None:
        cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',['white',color],256)

        cust_cmap._init()

        statistic = np.nan_to_num(statistic)

        if to_alpha:
            alphas = np.linspace(0, 1, cust_cmap.N+3)
            alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * 0.4
            cust_cmap._lut[:,-1] = alphas

    else:
        cust_cmap = matplotlib.colormaps[cmap]

    if fig is None:
        fig, ax = plt.subplots()



    plot = ax.imshow(statistic.T, 
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            aspect='auto', cmap=cust_cmap, origin='lower', vmin = vs[0], vmax = vs[1])

    # plt.imshow(statistic.T, interpolation='bicubic',
    #            interpolation_stage='rgba', origin='lower', 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap)
    
    if ret_stat:
        return fig, ax, plot, (statistic, x_edge, y_edge)

    return fig, ax, plot

def plot_2dhist(data_x, data_y, x_range, y_range, color, bins, fig = None, ax = None):
    density, _, _ = np.histogram2d(data_x, 
                                   data_y,
                                   bins = bins, density=True, 
                                   range=[x_range,y_range])
    
    density = density/np.max(density)

    if fig is None:
        fig, ax = plt.subplots(figsize = [6.6, 6.3])
    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',[color,color],256)
    
    cust_cmap._init()
    
    alphas = np.linspace(0, 1, cust_cmap.N+3)
    alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * 0.4
    cust_cmap._lut[:,-1] = alphas

    plot = ax.imshow(density.T, 
               extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               aspect='auto', cmap=cust_cmap, origin='lower', vmin = 0, vmax = 1)
    
    
    return fig, ax, plot

    # plt.imshow(density.T, interpolation='bicubic',
    #            interpolation_stage='rgba', origin='lower', 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap, vmin=0, vmax=1)
    

def nan_replacer(array):
    no_nan = np.argwhere(~np.isnan(array))
    tree = KDTree(no_nan)

    # Iterate over array where nans are present, and replace with nearest value
    for i, j in zip(*np.where(np.isnan(array))):
        query_point = np.array([[i, j]])
        _, nearest_index = tree.query(query_point)
        nearest_coord = no_nan[nearest_index][0]
        array[i, j] = array[nearest_coord[0], nearest_coord[1]]

    return array

def most_common_lab(lab):
    labs, counts = np.unique(lab, return_counts=True)
    return labs[np.argmax(counts)]


def overlay_plot(data_x, data_y, labels, statistic, cmap: str, alphas = None, bins = 30, x_range = [940, 960], y_range = [0.08,0.3], fig = None, ax = None):

    parent_map = matplotlib.colormaps[cmap]

    x_int_grid = np.linspace(min(data_x), max(data_x), 100)  # Adjust the number of points (100 here) as needed
    y_int_grid = np.linspace(min(data_y), max(data_y), 100)
    Xi, Yi = np.meshgrid(x_int_grid, y_int_grid)

    for i, label in enumerate(np.unique(labels)):
        mask = label == labels

        color = parent_map(label/(max(np.unique(labels)+1)))

        if label == -1:
            color = 'dimgrey'
        # interp = CloughTocher2DInterpolator((data_x[mask], data_y[mask]), alphas[mask], fill_value=np.nan)        
        # Z = interp(Xi,Yi)
        
        # Z = griddata((data_x[mask], data_y[mask]), alphas[mask],(Xi,Yi), method = 'linear')

        # Xi, Yi, Z = (Xi.flatten(), Yi.flatten(), Z.flatten())

        if alphas is None:
            bin_stat, xps, yps,_ = binned_statistic_2d(data_x[mask], data_y[mask], labels[mask], 
                                                  statistic = statistic, bins = bins, range=[x_range, y_range])
        else:
            bin_stat, xps, yps,_ = binned_statistic_2d(data_x[mask], data_y[mask], alphas[mask], 
                                                  statistic = statistic, bins = 20, range=[x_range, y_range])
        
        X,Y = np.meshgrid(xps[:-1],yps[:-1])
        

        cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',['white',color],256)

        cust_cmap._init()
        
        opacity = np.linspace(0, 1, cust_cmap.N+3)
        # opacity = np.heaviside(opacity - 0.5, np.ones_like(opacity)) * 0.4
        cust_cmap._lut[:,-1] = opacity

        bin_stat = bin_stat.flatten()
        # nan_mask = np.invert(np.isnan(bin_stat))
        bin_stat = np.nan_to_num(bin_stat)
        interp = CloughTocher2DInterpolator(list(zip(X.flatten(),Y.flatten())), bin_stat, fill_value = 0)
        
        Z = interp(Xi,Yi)

        fake_bin , _, _, _ = binned_statistic_2d(Xi.flatten(), Yi.flatten(), Z.flatten(), 
                                                  statistic = statistic, bins = bins, range=[x_range, y_range])

        plt.imshow(fake_bin, cmap = cust_cmap, extent=x_range + y_range, aspect='auto', origin='lower')
        plt.title(label)
    plt.show()


def all_cluster_plot(num_clusters, cluster_labels, data_x, data_y, cmap, bins, x_range, y_range, s = 1, fig = None, ax = None, 
                     show_cbar = True) -> tuple[matplotlib.figure.Figure, plt.Axes, object, object]:
        if fig is None:
            fig, ax = plt.subplots(figsize = [6.3,5])

        num_clusters = np.max(cluster_labels) + 1
        spect_map = matplotlib.colormaps[cmap]

        for i, label in enumerate(np.unique(cluster_labels, axis=0)):
            mask = cluster_labels  == i
            color = spect_map(((label + 0.5) / num_clusters))

            fig, ax, plot = plot_2dhist(data_x[mask],
                        data_y[mask],
                        x_range,
                        y_range,
                        color, bins, fig, ax
                        )
            
            ax.scatter(data_x[mask],
                        data_y[mask], color = color, zorder = 100, label = None, s = s)

        if show_cbar:

            
            
            norm = matplotlib.colors.Normalize(vmin=0, vmax=num_clusters)
            cbarmap = matplotlib.colormaps[cmap]
            my_cmap = cbarmap(np.arange(cbarmap.N))
            my_cmap[:,-1] = np.ones_like(my_cmap[:,-1])

            cbarmap = matplotlib.colors.ListedColormap(my_cmap)

            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cbarmap)

            cbar = fig.colorbar(mappable, ax=ax, boundaries = np.arange(stop = num_clusters + 1))
            tick_locs = (np.arange(num_clusters) + 0.5)*(num_clusters)/num_clusters
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(np.arange(num_clusters))

            return fig, ax, plot, cbar
        
        return fig, ax, plot, None


def plot_2d_stat_histv2(data_x, data_y, alphas, labels, fig=None, ax=None):
    """
    Note: append thermal cloud data to x, y, alpha and label, and give a label
    of -1

    Parameters
    ----------
    data_x : Wavelengths
        1D flattened array
    data_y : Pwrs
        1D flattened array
    alphas : probabilities
        1D flatten array
    labels : cluster labels
        1D flatten array


    Returns
    -------
    None.

    """



    x_range = (min(data_x), max(data_x))
    y_range=(min(data_y), max(data_y))



    statistic_label, _, _, _ = binned_statistic_2d(data_x, data_y,
                                             labels, bins=30,
                                             range=[x_range,y_range],
                                             statistic='mean')


    statistic_label = nan_replacer(statistic_label)


    x_grid = np.linspace(x_range[0], x_range[1], 100)  # Adjust the number of points (100 here) as needed
    y_grid = np.linspace(y_range[0], y_range[1]-1e-5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)


    #Handling the alphas
    #If this is weird, switch to method='nearest'. Maybe this would have been
    #an easier way to handle the nearest neightbour thing from the start!
    alphas_interp = griddata((data_x, data_y), alphas, (X, Y), method='nearest',
                             fill_value=(np.random.random(1)*0.2 + 0.8))

    X, Y, alphas_interp = X.flatten(), Y.flatten(), alphas_interp.flatten()

    #Alternative, but worse interpolation
    # alphas_interp=interp2d(data_x, data_y, alphas, kind='linear')
    # alphas_interp = alphas_interp(x_grid, y_grid)
    # alphas_interp=alphas_interp.flatten()

    statistic_alpha, _, _, _ = binned_statistic_2d(X, Y,
                                             alphas_interp, bins=30,
                                             range=[x_range,y_range])
    statistic_alpha[statistic_alpha>1] = 1
    statistic_alpha[statistic_alpha<0] = 0


    statistic_alpha = np.nan_to_num(statistic_alpha)


    if fig is None:
        fig, ax = plt.subplots()

    ax.imshow(statistic_label.T,
              extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
              aspect='auto', origin='lower', alpha=statistic_alpha.T)


def data_dict(files):
    powers = []
    lengths = []
    int_times = []
    images = []
    pcas = []
    for i, file in enumerate(files):
        
        split_file1 = file.split(sep)
        # print(split_file1)
        image = cv2.imread(file)


        split_file = split_file1[-1].split('_')
        # print(split_file)
        powers.append(float(split_file[4]))
        lengths.append(float(split_file[5]))
        int_times.append(float(split_file[3]))
        pcas.append(float(split_file[-2]))
        images.append(image.flatten())


    data = {
        'Files': np.array(files),
        'Powers': np.array(powers),
        'Lengths': np.array(lengths),
        'Int_times': np.array(int_times),
        'Pcas': np.array(pcas),
        'Images': np.array(images)
    }

    return data

if __name__ == '__main__':

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [6.3, 6.3],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pkl.load(f)

    # bec_crop_centre(r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\20240321\pbec_20240321_000118_7814.0_0.12576051724137932_941.7710571289062_7.275862068965518_.png",
    #                 files, (220,220), 
    #                 root = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321")


    stim_files, stim_mask = select_files.select_stimulated(files, 50)



    data = data_dict(files)

    data['PCA_Length'] = fit_pca(data['Lengths'], data['Pcas'])


    spect_map = matplotlib.cm.get_cmap('Spectral')

    label_files = np.array(glob('Apr_23_predicted_labels_*.pkl'))
    cluster_num = np.array([int(i.split('_')[-1][:-4]) for i in label_files])

    label_files = label_files[np.argsort(cluster_num)]

    with open('Apr_23_predicted_labels_7.pkl', 'rb') as f:
        cluster_labels = pkl.load(f)

    fig, ax, _, _ = all_cluster_plot(
        np.max(cluster_labels), cluster_labels,
        data['Lengths'][stim_mask], data['Powers'][stim_mask],
        'tab20b', 30, [940,960], [min(data['Powers']), max(data['Powers'])], 2
        )
    add_loss_rate(ax, 'inter.pkl')
    ax.set_xlabel('$\lambda$ ($nm$)')
    ax.set_ylabel('Pump Power (W)')
    plt.show()
    

    fig = plt.figure(figsize=[6.3, 5])


    fig, axes, gs = grid_plot(len(label_files), 4, 2, 0.1, 0.15,fig = fig)

    for i, file in enumerate(label_files):

        # with open('Apr_23_predicted_labels_7.pkl', 'rb') as f:
        #     cluster_labels = pkl.load(f)

        # with open('Apr_2_CNN_out.pkl', 'rb') as f:
        #     outs, preds = pkl.load(f)

        with open(file, 'rb') as f:
            cluster_labels = pkl.load(f)
        
        # cluster_labels, _ = Meta_classifier.quick_kmeans(data['Images'][stim_mask], 6)
        # cluster_labels = preds
        
        all_cluster_plot(
            np.max(cluster_labels), cluster_labels, 
            data['Lengths'][stim_mask], data['Powers'][stim_mask],
            'Spectral', 30, [940,960], [min(data['Powers']), max(data['Powers'])], fig, 
            axes[i])
        
        # add_loss_rate(axes[i], 'absfunc.pkl')
        axes[i].set_title(f'{np.sort(cluster_num)[i]} clusters')

    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\All_Kmeans.png')
    plt.show()





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

    # for i in ax.spines:
    #     ax.spines[i].set_color('w') 
    # ax.tick_params(color = 'w')
    # ax.yaxis.label.set_color('w')
    # ax.xaxis.label.set_color('w')
    plt.ylabel('Pump Power (W)')
    plt.xlabel('Cavity Length (nm)')

    # plt.tight_layout()

    plt.show()


    fig, axes = select_files.show_line_images(data['Files'][stim_mask],
                                line_files, 8,
                                cluster_labels, log = False, cmap='brg')

    plt.show()

    plt.scatter(data['Lengths'][np.invert(stim_mask)], data['Powers'][np.invert(stim_mask)], color = 'grey')
    plt.scatter(data['Lengths'][stim_mask], data['Powers'][stim_mask], c =cluster_labels, cmap='tab10')
    plt.show()