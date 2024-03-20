from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from scipy.ndimage import center_of_mass
import Meta_classifier
import matplotlib

files = glob(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\MultiMode Analysis\20240222\Cropped images\*.bmp')

# t_stamps = ['20240316_122615', '20240316_174943']

# Mar16_files = glob(r'C:\Users\Pouis\OneDrive - Imperial College London\20240316\*.png')

# files = []
# for i in Mar16_files:

#     if int(i.split('_')[2]) > int(t_stamps[0].split('_')[1]):
#         files.append(i)

# with open(r'MultiMode Analysis\relavent_files.pkl', 'wb') as f:
#     pkl.dump(files, f)


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
    #image = cv2.imread(file, 0)
    split_file1 = file.split(r'\\')
    print(split_file1)
    #plt.imshow(image
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
    'Files': np.array(files)[error_mask],
    'Powers': np.array(powers)[error_mask],
    'Lengths': np.array(lengths)[error_mask],
    'Int_times': np.array(int_times)[error_mask]
}

thermal_mask = data['Int_times'] > 1000

print(thermal_mask)

with open('MultiMode Analysis\predicted_labels', 'rb') as f:
    cluster_labels = pkl.load(f)


spect_map = matplotlib.cm.get_cmap('brg')

def plot_2dhist(data_x, data_y, x_range, y_range, color):
    density, _, _ = np.histogram2d(data_x, 
                                   data_y,
                                   bins = 12, density=True, 
                                   range=[x_range,y_range])
    
    density = density/np.max(density)

    
    cust_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap'+str(i),[color,color],256)
    
    cust_cmap._init()
    
    alphas = np.linspace(0, 1, cust_cmap.N+3)
    alphas = np.heaviside(alphas - 0.1, np.ones_like(alphas)) * 0.4
    cust_cmap._lut[:,-1] = alphas

    # plt.imshow(density.T, 
    #            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    #            aspect='auto', cmap=cust_cmap, origin='lower')

    plt.imshow(density.T, interpolation='bicubic',
               interpolation_stage='rgba', origin='lower', 
               extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               aspect='auto', cmap=cust_cmap)

phases = []
for i in np.unique(cluster_labels):
    cluster_idx = np.argwhere(cluster_labels == i)

    fig, axes = plt.subplots(nrows=2, ncols=3)


    count = 0 
    while count < 6:
        row = count // 3
        col = count % 3
        idx = np.random.randint(len(cluster_idx))
        image = cv2.imread(data['Files'][np.invert(thermal_mask)][cluster_idx[idx]][0], 0)
        axes[row,col].imshow(image)
        count += 1

    plt.show()
    in_phase = input("Enter Cluster Label: ")
    phases.append(in_phase)
    

plot_2dhist(data['Lengths'][thermal_mask], data['Powers'][thermal_mask],
            [min(data['Lengths']), max(data['Lengths'])],
            [min(data['Powers']), max(data['Powers'])],
            'grey')

plt.scatter(data['Lengths'][thermal_mask],
            data['Powers'][thermal_mask], color = 'grey', zorder = 100, label = 'Thermal Cloud')


for i in np.unique(cluster_labels):
    mask = cluster_labels  == i
    color = spect_map((i+1)/(max(np.unique(cluster_labels)+1)))

    plot_2dhist(data['Lengths'][np.invert(thermal_mask)][mask],
                data['Powers'][np.invert(thermal_mask)][mask],
                [min(data['Lengths']), max(data['Lengths'])],
                [min(data['Powers']), max(data['Powers'])],
                color
                )
    
    plt.scatter(data['Lengths'][np.invert(thermal_mask)][mask],
                data['Powers'][np.invert(thermal_mask)][mask], color = color, zorder = 100, label = phases[i])



plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.grid(c='black')
ax = plt.gca()
for i in ax.spines:
    ax.spines[i].set_color('w')
ax.tick_params(color = 'w')
ax.yaxis.label.set_color('w')
ax.xaxis.label.set_color('w')
plt.ylabel('Pump Power (W)')
plt.xlabel('Cavity Length (nm)')

plt.tight_layout()

plt.show()
plt.scatter(data['Lengths'][thermal_mask], data['Powers'][thermal_mask], color = 'grey')
plt.scatter(data['Lengths'][np.invert(thermal_mask)], data['Powers'][np.invert(thermal_mask)], c =cluster_labels,cmap='tab10')
plt.show()