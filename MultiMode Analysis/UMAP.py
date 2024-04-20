import cv2
from scipy.ndimage import zoom
import glob
import numpy as np
from scipy.ndimage import center_of_mass
from os.path import sep
from sklearn.decomposition import PCA
import cv2
from os.path import sep
import umap
import matplotlib.pyplot as plt
import scienceplots
from sklearn.cluster import KMeans
import tqdm
import gc




def read(bec_file: str, files: list[str]):
    img_features = []

    all_info = [file.split('_') for file in files]
    int_times = [float(file[4]) for file in all_info]
    bec_im = cv2.imread(bec_file, 0)
    cx, cy = center_of_mass(bec_im ** 3)

    for i, f in tqdm.tqdm(enumerate(files)):
        image = cv2.imread(f, 0)


        image = zoom(image, (224 / image.shape[0], 244 / image.shape[1]))

        size = (224, 224)
        image_crop = image[int(int(cx) - size[0] / 2):int(int(cx) + size[0] / 2),
                     int(int(cy) - size[1] / 2):int(int(cy) + size[1] / 2)]

        image = image.flatten()
        image = image/int_times[i]



        img_features.append(image)

    img_features = np.array(img_features)

    return img_features

plt.style.use(['science','ieee'])


def umap2d(neighbors, min_dist):
    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, random_state=22)
    embedding = reducer.fit_transform(img_features)

    kmeans = KMeans(n_clusters=10, random_state=22)
    kmeans.fit(img_features)
    
    # Plot
    plt.title(f'neighbours: {neighbors}, dist: {min_dist}')
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=kmeans.labels_, cmap='viridis')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

becpath = r'C:\Users\natak\OneDrive - Imperial College London\Documents\University\Year 4\MSci Project-r018104\AprData\Final_data\pbec_20240321_003143_62499.0_0.18152103448275864_940.0557861328125_9.0_.png'
path=r'C:\Users\natak\OneDrive - Imperial College London\Documents\University\Year 4\MSci Project-r018104\AprData\Final_data'
files = glob.glob(path+'\*.png')

img_features = read(becpath, files)

#Filtering
mask = np.max(img_features, axis=1) >= 50/500000
img_features = img_features[mask]

pca = PCA(n_components=100, random_state=22)
pca.fit(img_features)
img_features = pca.transform(img_features)


# #%%
# all_info = [file.split('_') for file in files]
# int_times = [int(file[4]) for file in all_info]

# #%%






#Neightbours
# neigh = np.linspace(4, 50, 10, dtype=int)
# for n in tqdm.tqdm(neigh):
#     umap2d(n, min_dist=0.1)
#     gc.collect()


if __name__ == '__main__':
    #Distances
    min_dists = np.linspace(0, 1, 11)
    for d in tqdm.tqdm(min_dists):
        umap2d(14, d)
        gc.collect()


#%%
# reducer = umap.UMAP(n_components=3)
# embedding = reducer.fit_transform(img_features)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5)
# ax.set_title('UMAP 3D projection')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_zlabel('UMAP 3')
# plt.show()
#%%


# neigh = np.linspace(4, 50, 5, dtype=int)
# min_dists = np.linspace(0, 1, 5)
# for n in tqdm.tqdm(neigh):
#     for d in min_dists:
#         umap2d(n, d)