from glob import glob
import numpy as np
import pickle as pkl
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

def open_img_files(t_stamps: list[str],  root_dir: str):

    mint = int(t_stamps[0].split('_')[0] + t_stamps[0].split('_')[1])
    maxt = int(t_stamps[1].split('_')[0] + t_stamps[1].split('_')[1])

    files = glob(root_dir + r'\\' + t_stamps[0].split('_')[0] + r'\*.png') + glob(root_dir + r'\\' + t_stamps[1].split('_')[0] + r'\*.png')

    rel_files = []

    for i in files:
        day = (i.split('\\')[-1].split('_')[1])
        time = (i.split('\\')[-1].split('_')[2])

        if mint <= int(day + time) <= maxt:
            rel_files.append(i)
    
    return np.unique(rel_files)

def select_stimulated(files, cuttoff: float):
    truth_list = []
    for i in tqdm(files):
        img = cv2.imread(i,0)

        truth_list.append(np.max(img) > cuttoff)
    print(truth_list)
    truth_list = np.array(truth_list)
    return files[truth_list], truth_list

def select_line(files, point: list[float], height: float):

    
    m = (point[0][1] - point[1][1]) / (point[0][0] - point[1][0])
    line = lambda x, xo, yo, m: x*m + yo - (xo*m)

    rel_files = []
    length_list = []
    for i in files:
        split_f = i.split(r'\\')
        split_n = split_f[-1].split('_')
        power = float(split_n[-4])
        length = float(split_n[-3])
        

        l_point = line(length,point[0][0], point[0][1], m) 

        if (l_point - height/2 < power < l_point + height/2) and (point[0][0] < length < point[1][0]):
            rel_files.append(i)
            length_list.append(length)

    return np.array(rel_files)[np.argsort(length_list)], line, [point[0][0], point[0][1], m], np.array(length_list)


def show_line_images(stim_files, line_files, col_length: int, cluster_list: list[int], cmap: str = 'brg', log = False):

    fig, axes = plt.subplots(nrows=len(line_files)//col_length + 1, ncols = len(line_files)//(len(line_files)//col_length))
    col_map = matplotlib.cm.get_cmap(cmap)
    for i, f in enumerate(line_files):

        arg = np.argwhere(stim_files == f)

        clust = cluster_list[arg]

        color = col_map(clust / (max(np.unique(cluster_list)) + 1))
        color = matplotlib.colors.to_hex(color[0][0])

        cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap'+str(i),['black',color],256)
    
        cust_cmap._init()


        row = i // col_length
        col = i % col_length
        im = cv2.imread(f,0)

        
        if log:
            im = np.log(im + 1)

        axes[row,col].imshow(im, cmap = cust_cmap)
        #axes[row,col].set_title(i)#
        axes[row,col].tick_params(tick1On = False,label1On = False, label2On = False, tick2On = False)


    return fig, axes

def plot_line(ax, line, line_params:list[float], height: float, points: list[list[float]]):

    x = np.linspace(points[0][0], points[1][0], 10)
    l1 = line(x, *line_params) - height/2
    ax.fill_between(x, l1, l1 + height, color = 'red', alpha = 0.5, zorder = 105)

    return ax


if __name__ == '__main__':
    t_stamps = ['20240320_213246', '20240321_141756']
    #root_dir = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files"
    
    root_dir = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321"
    # fs = open_img_files(t_stamps, root_dir)
    

    fs = np.array(glob(root_dir + r'\*.png'))

    rs, _ = select_stimulated(fs, 50)
    print(rs)

    with open(r'MultiMode Analysis\relavent_files.pkl', 'wb') as f:
        pkl.dump(fs ,f)
    
    with open(r'MultiMode Analysis\stim_files.pkl', 'wb') as f:
        pkl.dump(rs,f)

    
    line_files, line, line_params, _ = select_line(rs, [[945, 0.2], [950,0.12]], 0.05)

    

