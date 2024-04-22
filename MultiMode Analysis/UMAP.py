import cv2
from scipy.ndimage import zoom
import glob
import numpy as np
from scipy.ndimage import center_of_mass
from os.path import sep
import cv2
from os.path import sep
import umap
import matplotlib.pyplot as plt
import scienceplots
from sklearn.cluster import KMeans
import tqdm
import gc
import matplotlib
#import hdbscan
import UMAP_utils as uu
import pickle
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans



plt.style.use(['science','ieee'])


#You can supply your own method to get this info
becpath = r'C:\Users\natak\OneDrive - Imperial College London\Documents\University\Year 4\MSci Project-r018104\AprData\Final_data\pbec_20240321_003143_62499.0_0.18152103448275864_940.0557861328125_9.0_.png'
path=r'C:\Users\natak\OneDrive - Imperial College London\Documents\University\Year 4\MSci Project-r018104\AprData\Final_data'
files = glob.glob(path+'\*.png')

img_features, wavelengths, pwrs, thermal_wavelengths, thermal_pwrs = uu.load_raw_ims(becpath, files)
with open(r'C:/Users/natak/OneDrive - Imperial College London/Documents/University/Year 4/MSci Project-r018104/AprData/Apr_2001_features.pkl', 'rb') as f:
    img_features = pickle.load(f)


def umap2d(neighbors, min_dist, embedding_dim=2, kmeans_clusters=5):
    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, metric='euclidean', random_state=22, densmap=True, n_components=embedding_dim)
    embedding = reducer.fit_transform(img_features)

    kmeans = FuzzyKMeans(k=kmeans_clusters, m=2)
    kmeans.fit(img_features)
    cluster_labels = kmeans.labels_

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=30, cluster_selection_method='leaf')
    # cluster_labels = clusterer.fit_predict(img_features)

    # Plot
    plt.title(f'neighbours: {neighbors}, dist: {min_dist}')
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=cluster_labels, cmap='Spectral')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


    lab = embedding, cluster_labels
    return lab



if __name__ == '__main__':
    embedding, labels = umap2d(16, 0, embedding_dim=2, kmeans_clusters=8)

    #Check labels are the same length
    labels = labels[0:len(wavelengths)]

    #Plot binned plot. Replace with plot2dhistv2 at some point
    uu.resample_and_plot(np.vstack((wavelengths, pwrs, labels)).T, bins=(37, 30))

    #Cluster to umap
    kmeans = FuzzyKMeans(k=7, m=1.5)
    kmeans.fit(embedding)
    labels = kmeans.labels_

    #HDBSCAN is also avaliable in scikit learn, can be fuzzy,
    # kmeans = hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean', prediction_data='True', cluster_selection_method='eom')
    # labels = kmeans.fit_predict(img_features)
    #
    # z = hdbscan.all_points_membership_vectors(kmeans)
    # labels = np.argmax(z, axis=1)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=labels, cmap='Spectral')
    plt.title('K-means, umap')
    plt.show()

    uu.resample_and_plot(np.vstack((wavelengths, pwrs, labels)).T, bins=(37, 30))



    #Do you want an interactive plot?
    #Try uu.interactive_scatter(x, y, img_files), where x and y are the first and second embedding dims













    #Plot scatter plot of clustering before umap
    # scatter = plt.scatter(wavelengths, pwrs, c=labels)
    # plt.title('Clustering, no umap')
    # legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    # plt.gca().add_artist(legend1)
    # plt.show()

    # wavelengthst = np.append(wavelengths, thermal_wavelengths)
    # pwrst = np.append(pwrs, thermal_pwrs)
    # labelst = np.append(labels, np.ones(len(thermal_pwrs))*-1)
    #
    # t =np.vstack((wavelengthst, pwrst, labelst)).T
    # labels = labels[0:len(wavelengths)]
    # uu.resample_and_plot(t, bins=(37, 30))
    #
    # labels = labels[0:len(wavelengths)]
    # scatter = plt.scatter(wavelengths, pwrs, c=labels, cmap='Spectral')
    # plt.title('K-means, no umap')
    #
    # # Add legend
    # legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    # plt.gca().add_artist(legend1)
    # plt.show()



