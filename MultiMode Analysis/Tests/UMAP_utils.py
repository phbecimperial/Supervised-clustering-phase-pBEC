from scipy.stats import binned_statistic_2d
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import cv2
from scipy.ndimage import zoom
import glob
from scipy.ndimage import center_of_mass
from os.path import sep
import cv2
from os.path import sep
import scienceplots
from sklearn.cluster import KMeans
import tqdm
import gc
import matplotlib
import UMAP_utils as uu

spect_map = matplotlib.colormaps['brg']

def read(bec_file: str, files: list[str], features=True):
    img_features = []

    all_info = [file.split('_') for file in files]
    int_times = [float(file[4]) for file in all_info]
    pwrs = [float(file[5]) for file in all_info]
    wavelengths = [float(file[6]) for file in all_info]

    bec_im = cv2.imread(bec_file, 0)
    cx, cy = center_of_mass(bec_im ** 3)

    for i, f in tqdm.tqdm(enumerate(files)):
        image = cv2.imread(f, 0)


        image = zoom(image, (224 / image.shape[0], 244 / image.shape[1]))

        size = (224, 224)
        image_crop = image[int(int(cx) - size[0] / 2):int(int(cx) + size[0] / 2),
                     int(int(cy) - size[1] / 2):int(int(cy) + size[1] / 2)]

        if features == False:
            img_features.append(image)

        else:
            image = image.flatten()
            img_features.append(image)
            #image = image/int_times[i]





    img_features = np.array(img_features)

    return img_features, np.array(pwrs), np.array(wavelengths)

def load_raw_ims(becpath, files):
    img_features, pwrs, wavelengths = uu.read(becpath, files)

    # Filtering
    mask = np.max(img_features, axis=1) >= 50

    thermal_wavelengths = wavelengths[~mask]
    thermal_pwrs = pwrs[~mask]

    img_features = img_features[mask]
    pwrs = pwrs[mask]
    wavelengths = wavelengths[mask]

    pca = PCA(n_components=100, random_state=22)
    pca.fit(img_features)
    img_features = pca.transform(img_features)
    return img_features, wavelengths, pwrs, thermal_wavelengths, thermal_pwrs

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


def plot_2d_stat_hist(data_x, data_y, alphas, labels, fig=None, ax=None, single_plot=True):
    x_range = (min(data_x), max(data_x))
    y_range = (min(data_y), max(data_y))

    statistic_label, _, _, _ = binned_statistic_2d(data_x, data_y,
                                                   labels, bins=30,
                                                   range=[x_range, y_range],
                                                   statistic=most_common_lab)

    statistic_label = nan_replacer(statistic_label)

    x_grid = np.linspace(x_range[0], x_range[1], 100)  # Adjust the number of points (100 here) as needed
    y_grid = np.linspace(y_range[0], y_range[1] - 1e-5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Handling the alphas
    # If this is weird, switch to method='nearest'. Maybe this would have been
    # an easier way to handle the nearest neightbour thing from the start!
    alphas_interp = griddata((data_x, data_y), alphas, (X, Y), method='cubic',
                             fill_value=1, rescale=True)

    # alphas_interp=RBFInterpolator(np.column_stack((data_x, data_y)), alphas)
    X, Y, alphas_interp = X.flatten(), Y.flatten(), alphas_interp.flatten()

    # alphas_interp=interp2d(data_x, data_y, alphas, kind='linear')
    # alphas_interp = alphas_interp(x_grid, y_grid)
    # alphas_interp=alphas_interp.flatten()

    statistic_alpha, _, _, _ = binned_statistic_2d(X, Y,
                                                   alphas_interp, bins=30,
                                                   range=[x_range, y_range])
    statistic_alpha[statistic_alpha > 1] = 1
    statistic_alpha[statistic_alpha < 0] = 0

    statistic_alpha = np.nan_to_num(statistic_alpha)

    if fig is None:
        fig, ax = plt.subplots()

    return fig, ax


def resample_and_plot(data, bins):
    # data needs wavelengths, pwrs and labels

    # Determine grid dimensions
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    x_bins = np.linspace(x_min, x_max, bins[0] + 1)
    y_bins = np.linspace(y_min, y_max, bins[1] + 1)

    xlength = x_max - x_min
    ylength = y_max - y_min

    # Assign data points to grid boxes
    x_indices = np.digitize(data[:, 0], x_bins) - 1
    y_indices = np.digitize(data[:, 1], y_bins) - 1

    numlab = len(np.unique(data[:, 2]))

    # Determine the most common label for each grid box
    unique_indices = np.unique(np.column_stack((x_indices, y_indices)), axis=0)
    grid_data = np.zeros((len(unique_indices), 3), dtype=int)  # (x_index, y_index, most_common_label)
    for i, (x_idx, y_idx) in enumerate(unique_indices):
        points_in_box = data[(x_indices == x_idx) & (y_indices == y_idx)]
        labels, counts = np.unique(points_in_box[:, 2], return_counts=True)
        # print(counts)
        bin_label = labels[np.argmax(counts)]

        grid_data[i] = [x_idx, y_idx, bin_label]

    # Define colormap
    cmap = plt.get_cmap('Spectral', numlab)

    # Plot grid boxes
    fig, ax = plt.subplots()
    for x_idx, y_idx, label in grid_data:
        rect = plt.Rectangle((x_bins[x_idx], y_bins[y_idx]), xlength / bins[0], ylength / bins[1],
                             color=cmap(label / (numlab - 1)), alpha=1)
        ax.add_patch(rect)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title('Grid Boxes with Most Common Label')
    plt.grid(True)
    plt.show()

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from PIL import Image
import io
import base64

def interactive_scatter(x, y, img_files):
    masked_files = img_files
    images_base64 = []
    for file_name in masked_files:
        with open(file_name, "rb") as f:
            img = Image.open(f)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_base64.append(img_str)



    # Create ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        images=images_base64,
    ))

    # Create Bokeh plot
    plot = figure()

    # Add scatter plot
    plot.scatter(x='x', y='y', source=source, size=10)

    # Add HoverTool
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src='data:image/jpeg;base64, @{images}' style='width:200px; height:200px;'>
            </div>
        </div>
    """)
    plot.add_tools(hover)

    # Display plot
    show(plot)