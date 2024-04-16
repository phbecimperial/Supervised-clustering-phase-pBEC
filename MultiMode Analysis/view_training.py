import cv2
import pickle as pkl
from os.path import sep
import matplotlib.pyplot as plt

from glob import glob as glob

img_dir = r'C:\Users\Pouis\Documents\Uni Shit\Masters\Training Images'

im_files = glob(img_dir + sep + '*.pkl')

mode_tot = 0
for i in im_files:
    with open(i, 'rb') as f:
        img, label = pkl.load(f)
    
    # print(label)
    # plt.imshow(img)
    # plt.title(i)
    # plt.show()

    for i, item in enumerate(label):

        mode_tot += item[0]

print(mode_tot / (len(im_files) * 8))