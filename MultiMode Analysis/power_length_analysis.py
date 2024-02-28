from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass
import Meta_classifier

files = glob(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\MultiMode Analysis\20240222\Cropped images\*')

def crop_save_image(image,size,name,root):
    cx, cy  = center_of_mass(image**5)
    
    image_crop = image[int(int(cx) - size[0]/2):int(int(cx) + size[0]/2),
          int(int(cy) - size[1]/2):int(int(cy) + size[1]/2)]
    print(int(int(cx) - size[0]/2),int(int(cx) + size[0]/2))
    print(int(int(cy) - size[0]/2),int(int(cy) + size[0]/2))
    print(cx,cy)

    flag =  cv2.imwrite(root + r"\\" + name + '.bmp', image_crop)
    print(root + '/' + name + '.bmp')
    ##return image_crop

    

powers = []
lengths = []
int_times = []
images = []
for i, file in enumerate(files):
    image = cv2.imread(file, 0)
    split_file1 = file.split('\\')
    print(split_file1)
    #plt.imshow(image)
    #plt.show()
    #crop_save_image(image, (180,180), 'pbecCrop' + split_file1[1][3:-4], r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\MultiMode Analysis\20240222')
    #plt.imshow(image)
    #'plt.show()
    split_file = file.split('_')
    powers.append(float(split_file[4]))
    lengths.append(float(split_file[5]))
    int_times.append(float(split_file[3]))
    #images.append(image)

error_mask = np.array(lengths) < 952

data = {
    #'Images': np.array(images)[error_mask],
    'Powers': np.array(powers)[error_mask],
    'Lengths': np.array(lengths)[error_mask],
    'Int_times': np.array(int_times)[error_mask]
}

thermal_mask = data['Int_times'] == max(data['Int_times'])

print(thermal_mask)

plt.scatter(data['Lengths'][thermal_mask], data['Powers'][thermal_mask], color = 'grey')
plt.scatter(data['Lengths'][np.invert(thermal_mask)], data['Powers'][np.invert(thermal_mask)])
plt.show()