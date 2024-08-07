import matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob
import pickle as pkl


with open(r"C:\Users\Pouis\Documents\Uni Shit\Masters\Training Images\training_image@1712690355.7169082@010000.pkl", 'rb') as f:
    training_image, label = pkl.load(f)

expimg = Image.open(r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\20240321\pbec_20240321_074528_3907.0_0.16631362068965516_940.0557861328125_-3.0_.png")

plt.imshow(training_image)
plt.show()

plt.imshow(expimg)
plt.show()