import sys

sys.path.append('imports')
import pickle as pkl
import numpy as np
import json
from os.path import sep
import cv2
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import select_files
import scienceplots
from scipy.interpolate import interp1d
from generate_training import modelist
import power_length_analysis
import Meta_classifier
import socket

plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    'figure.figsize': [7.2, 7.2],
    'font.size': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300
})


def run_all(t_chunks, sys_path, crop_path, bec_path, position_index=None, threshold=1000):
    image_dir = sys_path
    #Make new folder or everything will be deleted
    crop_dir = sys_path + crop_path
    files, metas = select_files.open_img_files(t_chunks, image_dir)

    bec_file = sys_path + bec_path
    #print(bec_file)
    print(len(files))
    power_length_analysis.bec_crop_centre_fast(bec_file, files, [170, 170], crop_dir, plot=False)

    files = np.array(glob(crop_dir + sep + '*.png'))
    #print(len(files))
    data = power_length_analysis.data_from_metas_fast(metas, files)

    stim_mask = select_files.select_stimulated_exp_from_filename(data['file'], threshold)


    # Fit PCA values to cavity length
    if 'pca' not in data.keys():
        print('!')
        png_data = power_length_analysis.data_dict(files)
        data.update({'pca': png_data['pca']})

    pca_length = []

    for i in t_chunks:
        # Creates mask for stimulated emission and each time cavity is locked
        mint = int(i[0].split('_')[0] + i[0].split('_')[1])
        maxt = int(i[1].split('_')[0] + i[1].split('_')[1])
        #print(maxt)
        tmask = (mint <= data['t']) & (data['t'] <= maxt)
        mask = tmask & stim_mask

        # plt.scatter(data['cavity_length'][mask], data['power'][mask])
        # plt.show()

        mean_length = np.median(data['cavity_length'][mask])
        #print(mean_length)
        length_mask = (mean_length - 10 <= data['cavity_length']) & (data['cavity_length'] <= mean_length + 10)
        mask = tmask & stim_mask & length_mask

        #plt.scatter(data['pca'][mask], data['cavity_length'][mask])
        # plt.show()
        # fits pca values to cavity length
        _, fit = power_length_analysis.fit_pca(data['cavity_length'][mask], data['pca'][mask])
        pca_length += list(fit(data['pca'][tmask]))

        x = np.linspace(min(data['pca'][mask]), max(data['pca'][mask]), 20)
        # plt.plot(x, fit(x))
        # plt.title(i)
        # plt.show()

    data.update({'pca_length': np.array(pca_length)})

    # plt.scatter(data['pca_length'], data['cavity_length'])
    # plt.show()


    if 'position' in data.keys():
        position = np.unique(data['position'])[position_index]
        pos_mask = select_files.select_position(data, position)
        mask = stim_mask & pos_mask

    return data, mask

