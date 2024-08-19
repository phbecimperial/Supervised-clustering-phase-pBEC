import os
from os.path import sep
from glob import glob
import matplotlib.colors
import matplotlib.figure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import json
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
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed



def crop_save_image(files,size,root):
    """Crops around the center of an image and saves it


    Args:
        files (List[str]): List of files to crop
        size (tuple[int,int]): Dimensions of cropped image
        root (str): Path to save images
    """    



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


def add_loss_rate(ax: plt.Axes, abs_path) -> callable:
    """Fits and adds loss rate to a given axis

    Args:
        ax (plt.Axes): axes to add thermalisation to
        abs_path (str): path with interpolated absorption rates
    """

    with open(abs_path, 'rb') as f:
        abs_rate = pkl.load(f)

    llim, rlim = ax.get_xlim()
    x = np.linspace(llim, rlim, 2)

    # thermal = absorb(x)/70
    xto_thermal = lambda y:  abs_rate(y*1e-9)/8e10



    thermalto_x = interp1d(xto_thermal(x), x, fill_value='extrapolate')
    ax.secondary_xaxis('top', functions=(xto_thermal, thermalto_x), xscale = 'log', xlabel = '$\gamma$')

    return xto_thermal


def bec_crop_centre_loop(f, cx, cy, size: list[int, int], root: str):
    image = cv2.imread(f, 0)
    if image is None:
        print(f)
    name = f.split(sep)[-1][:-4]

    image_crop = image[int(int(cx) - size[0]/2):int(int(cx) + size[0]/2),
        int(int(cy) - size[1]/2):int(int(cy) + size[1]/2)]
    #print(int(int(cx) - size[0]/2),int(int(cx) + size[0]/2))
    #print(int(int(cy) - size[0]/2),int(int(cy) + size[0]/2))
    #print(cx,cy)
    # image_crop = sigmoid(image_crop, 0.01, 0.1)*255
    name = root + r"\\" + 'Crop' + name + '.png'
    if len(name) > 259:
        warnings.warn('PATH NAME TOO LONG. FILENAME HAS BEEN AUTOMATICALLY SHORTENED!')
        name = name[:-(len(name) - 259 + 4)] + '.png'

    flag =  cv2.imwrite(name, image_crop)


    

def bec_crop_centre(bec_file: str, files: list[str], size: list[int,int], root: str, plot=True):
    """    New image cropping fn, parse in file of bec image and will take as center for all other images.
    V simple don't know why I didn't think of this before

    Args:
        bec_file (str): File name for image of bec
        files (list[str]): list of files to crop
        size (list[int,int]): dimensions of cropeed image
        root (str): directory to save images
    """
    bec_im = cv2.imread(bec_file, 0)

    # cx, cy = center_of_mass(bec_im**8)
    ravel_arg = np.argmax(bec_im)
    cx, cy = np.unravel_index(ravel_arg, bec_im.shape)
    bec_im[:int(cx - size[0]/2) ,:int(cy - size[1]/2)] = 0
    bec_im[int(cx + size[0]/2):, int(cy + size[1]/2):] = 0

    # cx, cy = center_of_mass(bec_im**8)

    from Meta_classifier import sigmoid
    if plot:
        plt.imshow(bec_im)
        plt.show()

    bec_im = sigmoid((bec_im- bec_im.min())/(bec_im.max()-bec_im.min()), 0.05, 0.95)

    if plot:
        plt.imshow(bec_im)
        plt.show()

    cx,cy = center_of_mass(bec_im)

    for f in glob(root + sep + '*'):
        os.remove(f)

    #cropped_files = []
    #fnames = []

    for f in tqdm(files, leave=True):
        image = cv2.imread(f, 0)
        if image is None:
            print(f)
        name = f.split(sep)[-1][:-4]



        
        image_crop = image[int(int(cx) - size[0]/2):int(int(cx) + size[0]/2),
            int(int(cy) - size[1]/2):int(int(cy) + size[1]/2)]
        #print(int(int(cx) - size[0]/2),int(int(cx) + size[0]/2))
        #print(int(int(cy) - size[0]/2),int(int(cy) + size[0]/2))
        #print(cx,cy)
        # image_crop = sigmoid(image_crop, 0.01, 0.1)*255
        name = root + r"\\" + 'Crop' + name + '.png'
        if len(name) > 259:
            warnings.warn('PATH NAME TOO LONG. FILENAME HAS BEEN AUTOMATICALLY SHORTENED!')
            name = name[:-(len(name) - 259 + 4)] + '.png'

        flag =  cv2.imwrite(name, image_crop)
        #cropped_files.append(len(np.array(glob(root + sep + '*.png')))) 
        #fnames.append(root + r"\\" + 'Crop' + name + '.png')
        #print(root + r'\\' + name + '.png')
    print('completed crop')

def bec_crop_centre_fast(bec_file: str, files: list[str], size: list[int,int], root: str, plot=True):
    """    New image cropping fn, parse in file of bec image and will take as center for all other images.
    V simple don't know why I didn't think of this before

    Args:
        bec_file (str): File name for image of bec
        files (list[str]): list of files to crop
        size (list[int,int]): dimensions of cropeed image
        root (str): directory to save images
    """
    bec_im = cv2.imread(bec_file, 0)

    # cx, cy = center_of_mass(bec_im**8)
    ravel_arg = np.argmax(bec_im)
    cx, cy = np.unravel_index(ravel_arg, bec_im.shape)
    bec_im[:int(cx - size[0]/2) ,:int(cy - size[1]/2)] = 0
    bec_im[int(cx + size[0]/2):, int(cy + size[1]/2):] = 0

    # cx, cy = center_of_mass(bec_im**8)

    from Meta_classifier import sigmoid
    if plot:
        plt.imshow(bec_im)
        plt.show()

    bec_im = sigmoid((bec_im- bec_im.min())/(bec_im.max()-bec_im.min()), 0.05, 0.95)

    if plot:
        plt.imshow(bec_im)
        plt.show()

    cx,cy = center_of_mass(bec_im)

    for f in glob(root + sep + '*'):
        os.remove(f)

    #cropped_files = []
    #fnames = []

    Parallel(n_jobs=-1, verbose=1)(delayed(bec_crop_centre_loop)(f, cx, cy, size, root) for f in files)

    print('completed crop')

def fit_pca(lengths: npt.ArrayLike, pcas: npt.ArrayLike):
    """Fits measured cavity lengths to PCA values

    Args:
        lengths (npt.ArrayLike): Cavity Lengths
        pcas (npt.ArrayLike): pca values

    Returns:
        npt.ArrayLike : fitted values
    """
    
    # plt.scatter(pcas, lengths)
    # plt.show()
    popt = np.polyfit(pcas, lengths, 3)
    fit = np.poly1d(popt)

    return fit(pcas), fit


def grid_plot(nplots ,ncols, nrows, wspace, hspace = 0.4, tick_spacing = 10, fig = None):
    """Creates a grid of axes, can handle odd numbers of rows and computes spacing for remainder.

    Args:
        nplots (_type_): _description_
        ncols (_type_): _description_
        nrows (_type_): _description_
        wspace (_type_): _description_
        hspace (float, optional): _description_. Defaults to 0.4.
        tick_spacing (int, optional): _description_. Defaults to 10.
        fig (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

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

def plot_2d_stat_hist(data_x, data_y, alphas, x_range, y_range, bins = 10,
                      color = None, cmap = None, fig = None, ax = None, to_alpha = False,
                      vs = [0,1], ret_stat = False, facecolor='darkgrey'):

    # density, xedges, yedges = np.histogram2d(data_x, 
    #                                data_y,
    #                                bins = 12, density=True, 
    #                                range=[x_range,y_range])
    
    
    statistic, x_edge, y_edge ,_ = binned_statistic_2d(data_x, data_y, 
                                         alphas, bins=bins, 
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



    # plot = ax.imshow(statistic.T, 
    #         extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #         aspect='auto', cmap=cust_cmap, origin='lower', vmin = vs[0], vmax = vs[1])

    plot = ax.pcolor(bins[0], bins[1], statistic.T, cmap=cust_cmap, vmin = vs[0], vmax = vs[1])
    ax.set_facecolor(facecolor)

    # plt.imshow(statistic.T, interpolation='bicubic',
    #            interpolation_stage='rgba', origin='lower', 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap)
    
    
    if ret_stat:
        return fig, ax, plot, (statistic, x_edge, y_edge)

    return fig, ax, plot

def plot_2dhist(data_x, data_y, x_range, y_range, color, bins, fig = None, ax = None, bin_edges = None, alpha = 0.4):

    if bin_edges is None:
        density, _, _ = np.histogram2d(data_x, 
                                    data_y,
                                    bins = bins, density=True, 
                                    range=[x_range,y_range])
    else:
        density, _, _ = np.histogram2d(data_x, 
                            data_y,
                            bins = bin_edges, density=True, 
                            range=[x_range,y_range])
    
    density = density/np.max(density)

    if fig is None:
        fig, ax = plt.subplots(figsize = [6.6, 6.3])
    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',[color,color],256)
    
    cust_cmap._init()
    
    alphas = np.linspace(0, 1, cust_cmap.N+3)
    alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * alpha
    cust_cmap._lut[:,-1] = alphas

    plot = ax.imshow(density.T, 
               extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               aspect='auto', cmap=cust_cmap, origin='lower', vmin = 0, vmax = 1)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    
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
        # plt.title(label)
    plt.show()


def all_cluster_plot(num_clusters, cluster_labels, data_x, data_y, cmap, bins, x_range, y_range, s = 1, fig = None, ax = None, 
                     show_cbar = True, bin_edges = None) -> tuple[matplotlib.figure.Figure, plt.Axes, object, object]:
        if fig is None:
            fig, ax = plt.subplots(figsize = [6.3,5])

        num_clusters = np.max(cluster_labels) + 1 
        spect_map = matplotlib.colormaps[cmap]

        for i, label in enumerate(np.unique(cluster_labels, axis=0)):
            mask = cluster_labels  == i
            color = spect_map(((label + 0.5) / (num_clusters + 0.5)))
            fig, ax, plot = plot_2dhist(data_x[mask],
                        data_y[mask],
                        x_range,
                        y_range,
                        color, bins, fig, ax, bin_edges = bin_edges
                        )
            
            ax.scatter(data_x[mask],
                        data_y[mask], color = color, zorder = 100, label = None, s = s)
            #ax.set_yscale('log')
            #ax.set_xscale('log')

        if show_cbar:

            
            
            norm = matplotlib.colors.Normalize(vmin=0, vmax=num_clusters + 0.5)
            cbarmap = matplotlib.colormaps[cmap]
            my_cmap = cbarmap(np.arange(cbarmap.N))
            my_cmap[:,-1] = np.ones_like(my_cmap[:,-1])

            cbarmap = matplotlib.colors.ListedColormap(my_cmap)

            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cbarmap)

            cbar = fig.colorbar(mappable, ax=ax, boundaries = np.arange(0, stop = num_clusters + 0.5))
            tick_locs = (np.arange(0, num_clusters) + 0.5)
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

def data_from_metas(metas, files):
    from generate_training import quick_norm
    param_dict = {}
    param_dict.update({'t': []})
    param_dict.update({'image': []})
    param_dict.update({'flat_image': []})
    for i, (meta,file) in enumerate(zip(metas, files)):
        with open(meta, 'r') as f:
            meta = json.load(f)
        for j, (key, val) in enumerate(meta['parameters'].items()):
            if key not in param_dict.keys():
                param_dict.update({key: []})
            
            param_dict[key].append(val)
        
        im = cv2.imread(file, 0)
        t = meta['ts'].split('_')[0] + meta['ts'].split('_')[1]
        param_dict['t'].append(int(t))
        param_dict['image'].append(im)
        param_dict['flat_image'].append(quick_norm(im).flatten())

        

    
    param_dict.update({'file': files})

    data = {}
    for key, val in param_dict.items():
        data.update({key: np.array(val)})
    return data

from generate_training import quick_norm
def process_file(meta_file, image_file):
    with open(meta_file, 'r') as f:
        meta = json.load(f)    
    data = {}

    for key, val in meta['parameters'].items():
        data[key] = val
    
    im = cv2.imread(image_file, 0)
    t = meta['ts'].split('_')[0] + meta['ts'].split('_')[1]
    data['t'] = int(t)
    data['image'] = im
    data['flat_image'] = quick_norm(im).flatten()
    return data

def data_from_metas_fast(metas, files):
    results = Parallel(n_jobs=-1)(delayed(process_file)(meta, file) for meta, file in zip(metas, files))

    param_dict = {key: [] for key in results[0].keys()}
    param_dict.update({'file': []})

    for result, file in zip(results, files):
        for key, val in result.items():
            if key not in param_dict:
                param_dict[key] = []
            param_dict[key].append(val)
        param_dict['file'].append(file)

    data = {key: np.array(val) for key, val in param_dict.items()}
    
    return data
    
    
    




def data_dict(files):
    powers = []
    lengths = []
    int_times = []
    images = []
    pcas = []
    for i, file in tqdm.tqdm(enumerate(files)):
        
        split_file1 = file.split(sep)
        # print(split_file1)
        image = cv2.imread(file, 0)


        split_file = split_file1[-1].split('_')
        # print(split_file)
        powers.append(float(split_file[4]))
        lengths.append(float(split_file[5]))
        int_times.append(float(split_file[3]))
        pcas.append(float(split_file[-2]))
        images.append(image.flatten())


    data = {
        'file': np.array(files),
        'power': np.array(powers),
        'length': np.array(lengths),
        'int_time': np.array(int_times),
        'pca': np.array(pcas),
        'image': np.array(images)
    }

    return data

if __name__ == '__main__':

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [7.2, 6],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    with open('inter.pkl', 'rb') as f:
        abr = pkl.load(f)
    


    with open(r'MultiMode Analysis\relavent_files.pkl', 'rb') as f:
        files = pkl.load(f)

    # bec_crop_centre(r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\20240321\pbec_20240321_000118_7814.0_0.12576051724137932_941.7710571289062_7.275862068965518_.png",
    #                 files, (220,220), 
    #                 root = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321")


    stim_files, stim_mask = select_files.select_stimulated(files, 50)



    data = data_dict(files)

    data['PCA_Length'] = fit_pca(data['Lengths'], data['Pcas'])


    label_files = np.array(glob('Apr_26_predicted_labels_*.pkl'))
    cluster_num = np.array([int(i.split('_')[-1][:-4]) for i in label_files])

    label_files = label_files[np.argsort(cluster_num)]
    cluster_num = np.sort(cluster_num) - 3
    # with open('Apr_26_predicted_labels_8.pkl', 'rb') as f:
    #     cluster_labels = pkl.load(f)


    cluster_labels,_ = Meta_classifier.Kmeans_no_CNN(data['Files'][stim_mask], 7)

    fig, ax, _, _ = all_cluster_plot(
        np.max(cluster_labels), cluster_labels,
        data['Lengths'][stim_mask], data['Powers'][stim_mask],
        'tab20b', 30, [940,960], [min(data['Powers']), max(data['Powers'])], 8
        )
    
    fig.set_figwidth(3.7)
    fig.set_figheight(2.5)

    add_loss_rate(ax, 'inter.pkl')
    ax.set_xlabel('$\lambda$ ($nm$)')
    ax.set_ylabel('Pump Power (W)')
    # plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\Kmeans plots\Just_Kmeans_7.pdf', format = 'pdf')
    plt.show()
    

    fig = plt.figure(figsize=[7.2, 6])


    fig, axes, gs = grid_plot(len(label_files), 3, 2, 0.1, 0.15,fig = fig)

    for i, file in enumerate(label_files):

        # with open('Apr_23_predicted_labels_7.pkl', 'rb') as f:
        #     cluster_labels = pkl.load(f)

        # with open('Apr_2_CNN_out.pkl', 'rb') as f:
        #     outs, preds = pkl.load(f)

        # with open(file, 'rb') as f:
        #     cluster_labels = pkl.load(f)
        
        cluster_labels, _ = Meta_classifier.Kmeans_no_CNN(data['Files'][stim_mask], cluster_num[i])
        # cluster_labels = preds
        
        all_cluster_plot(
            np.max(cluster_labels), cluster_labels, 
            data['Lengths'][stim_mask], data['Powers'][stim_mask],
            'tab20b', 30, [940,960], [min(data['Powers']), max(data['Powers'])], fig = fig, ax = axes[i])
        
        # add_loss_rate(axes[i], 'absfunc.pkl')
        axes[i].set_title(f'{np.sort(cluster_num)[i]} clusters')

    fig.supxlabel('$\lambda$ ($nm$)')
    fig.supylabel('Pump Power (W)')
    plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\Kmeans plots\All_Just_Kmeans.pdf', format = 'pdf')
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

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from PIL import Image
import io
import base64
from bokeh.transform import linear_cmap
from bokeh.palettes import Category10

def interactive_scatter(x, y, img_files, labels_data):
    masked_files = img_files
    images_base64 = []
    for file_name in masked_files:
        with open(file_name, "rb") as f:
            img = Image.open(f)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_base64.append(img_str)


    num_labels = len(set(labels_data))
    palette = Category10[num_labels]


    # Create ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        images=images_base64,
        filename=masked_files,
        labels=labels_data
    ))

    # Create Bokeh plot
    plot = figure()

    # Add scatter plot
    plot.scatter(x='x', y='y', source=source, size=10, fill_color=linear_cmap('labels', palette, min(labels_data), max(labels_data)))

    # Add HoverTool
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src='data:image/jpeg;base64, @{images}' style='width:200px; height:200px;'>
            </div>
            <div>
                <span>@filename</span>
            </div>
        </div>
    """)
    plot.add_tools(hover)

    # Display plot
    show(plot)