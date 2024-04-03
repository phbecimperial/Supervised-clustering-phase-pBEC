import cv2
import pickle as pkl
from os.path import sep
import matplotlib.pyplot as plt

from glob import glob as glob

img_dir = r'C:\Users\Pouis\OneDrive - Imperial College London\202403_link\Training_Images'

im_files = glob(img_dir + sep + '*.pkl')

for i in im_files:
    with open(i, 'rb') as f:
        img = pkl.load(f)[0]
    
    plt.imshow(img)
    plt.title(i)
    plt.show()