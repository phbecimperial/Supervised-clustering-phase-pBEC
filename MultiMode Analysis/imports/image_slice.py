import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import sep
# "C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321\Croppbec_20240320_225530_500000.0_0.07_946.6923828125_3.4827586206896557_.png"
files = glob(r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321" + sep + r"*3.4827586206896557*")

for i in files[:7]:
    im = cv2.imread(i,0)
    im = im/np.max(im)
    plt.plot(im[112], label = i.split('_')[-4])

plt.legend()
plt.show()