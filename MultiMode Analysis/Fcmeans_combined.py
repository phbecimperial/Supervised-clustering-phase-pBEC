import matplotlib.pyplot as plt
import torch
import numpy as np
# from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
#from fcmeans import FCM
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from tqdm import tqdm
from pickle_Dataset import CustomDataset
from Fcmeans_utils import CustomModel, CombinedModel, custom_accuracy, match_rows
import torch
import torch.nn as nn


models = []
clusters = 5

for i in range(0, clusters):
    model = torch.load("Models/Res_Class_{}.pt".format(i))
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    newmodel = CustomModel(model)
    models.append(newmodel)

new_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model = CombinedModel(models)

test_dataset = CustomDataset(root_dir='Training_images_2', new_size=(224, 224))
#sampler = RandomSampler(test_dataset, num_samples=1000, replacement=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    model.eval()

    all_features = []
    all_labels = []
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)

        outputs = combined_model(inputs)
        outputs = outputs.cpu().numpy()
        outputs = outputs[0].tolist()
        all_features.append(outputs)  # Test
        all_labels.append(labels)

all_features = np.array(all_features)

# Insert PCA if needed
target_names = test_dataset.classes

# print("INFO: Starting clustering")
fcm = FuzzyKMeans(k=clusters, m=1.5)
fcm.fit(all_features)
fuzzy_membership_matrix = fcm.fuzzy_labels_
fuzzy_membership_matrix = fuzzy_membership_matrix.T

# Figure out which pieces of data are at what point, labelled by the indices.
# groups = {}
# for i in range(0, len(labels)):
#     cluster = labels[i]
#     if labels[i] not in groups.keys():
#         groups[cluster] = []
#         groups[cluster].append(i)
#     else:
#         groups[cluster].append(i)

# print(fuzzy_membership_matrix)
# Getting unique labels
#u_labels = np.unique(labels)

from sklearn.metrics import accuracy_score, classification_report

# Assuming you have true labels
targets = test_dataset.targets
array_of_arrays = np.array([[int(char) for char in string] for string in targets])
true_lab = array_of_arrays
true_lab = true_lab.T
true_lab = true_lab[:clusters]






matched_indices = match_rows(true_lab, fuzzy_membership_matrix)
fuzzy_membership_matrix = fuzzy_membership_matrix[matched_indices]

print("Matched indices:", matched_indices)


from sklearn.decomposition import PCA

# # Initialise
# pca = PCA(n_components=2)
# pca.fit(fuzzy_membership_matrix)
#
# # Explained variance ratio
# print("Explained variance ratio:", pca.explained_variance_ratio_)
#
# # Principal components
# print("Principal components:", pca.components_)
#
# result = pca.transform(fuzzy_membership_matrix)
#
# # Assuming X_pca has shape (n_samples, 2)
# plt.scatter(result[:, 0], result[:, 1])
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA')
#
# # Initialise
# pca = PCA(n_components=2)
# pca.fit(fuzzy_membership_matrix)
#
# # Explained variance ratio
# print("Explained variance ratio:", pca.explained_variance_ratio_)
#
# # Principal components
# print("Principal components:", pca.components_)
#
# result = pca.transform(fuzzy_membership_matrix)
#
# # Assuming X_pca has shape (n_samples, 2)
# plt.scatter(result[:, 0], result[:, 1])
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA')
#
#
# plt.show()