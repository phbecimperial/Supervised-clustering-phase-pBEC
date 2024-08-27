import numpy as np
import json
import cv2
from glob import glob
from os.path import sep

def open_img_files(t_stamps_list: list[list[str]], root_dir: str):
    rel_files = []
    rel_metas = []

    for t_stamps in t_stamps_list:

        mint = int(t_stamps[0].split('_')[0] + t_stamps[0].split('_')[1])
        maxt = int(t_stamps[1].split('_')[0] + t_stamps[1].split('_')[1])
 
        files = glob(root_dir + sep + '*' + t_stamps[0].split('_')[0] + '*' + sep + '*.png') + glob(
            root_dir + sep + '*' + t_stamps[1].split('_')[0] + '*' + sep + '*.png')
        metas = glob(root_dir + sep + t_stamps[0].split('_')[0] + sep + '*meta.json') + glob(
            root_dir + sep + t_stamps[1].split('_')[0] + sep + '*meta.json')

        # files = glob(root_dir + sep + '**' + sep + '*' + t_stamps[0].split('_')[0] + '*' + '.png', recursive = True) + glob(
        #     root_dir + sep + '**' + sep + '*' + t_stamps[1].split('_')[0] + '*' + '.png', recursive = True)
        # metas = glob(root_dir + sep + '**' + sep + '*' + t_stamps[0].split('_')[0] + '*' + '.json', recursive = True) + glob(
        #     root_dir + sep + '**' + sep + '*' + t_stamps[0].split('_')[1] + '*' + '.json', recursive = True)


        for i, file in enumerate(files):
            day = (file.split(sep)[-1].split('_')[1])
            time = (file.split(sep)[-1].split('_')[2])

            if mint <= int(day + time) <= maxt:
                rel_files.append(file)

        for i, meta in enumerate(metas):
            day = (meta.split(sep)[-1].split('_')[1])
            time = (meta.split(sep)[-1].split('_')[2])

            if mint <= int(day + time) <= maxt:
                rel_metas.append(meta)

    rel_files, args = np.unique(rel_files, return_index=True)
    rel_metas, args = np.unique(rel_metas, return_index=True)

    # for meta, file in rel_metas, rel_files:

    # rel_metas = list(np.array(rel_metas)[args])
    # print(len(rel_files))
    # print(len(rel_metas))
    return rel_files, rel_metas


def quick_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def data_from_metas(metas, files):
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
