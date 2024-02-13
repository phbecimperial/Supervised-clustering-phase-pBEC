import matplotlib.pyplot as plt
import torch
import numpy as np
# from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from fcmeans import FCM
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from tqdm import tqdm
from pickle_Dataset import CustomDataset

class CustomModel(torch.nn.Module):
    def __init__(self, original_model):
        super(CustomModel, self).__init__()
        # Get all layers except the last one
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        # Add your linear layer with the appropriate input size
        self.fc = torch.nn.Sequential(list(original_model.children())[-2])

    def forward(self, x):
        features_output = self.features(x)
        # Flatten the output before passing it to the linear layer
        flattened_output = features_output.view(features_output.size(0), -1)
        linear_output = self.fc(flattened_output)
        # Add any additional processing or layers if needed
        return linear_output

def view_cluster(cluster):
    indices = groups[cluster]
    if len(indices) > 30:
        indices = indices[:29]

    plt.figure(figsize=[25, 25])

    for i in range(0, len(indices)):
        plt.subplot(10, 10, i + 1)
        img = test_dataset[indices[i]][0].cpu().numpy()
        img = img[0]
        plt.imshow(img)

new_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("Models/Res_Class_2", map_location=torch.device('cpu'))
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel = CustomModel(model)


test_dataset = CustomDataset(root_dir='Training_images', new_size=(224, 224))
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    model.eval()

    all_features = []

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)

        outputs = newmodel(inputs)
        outputs = outputs.cpu().numpy()
        outputs = outputs[0].tolist()
        all_features.append(outputs) #Test

all_features = np.array(all_features)

# Insert PCA if needed

from sklearn.decomposition import PCA

#all_features = PCA(512).fit_transform(all_features)

target_names = test_dataset.classes

#%%
print("INFO: Starting clustering")
fcm = FuzzyKMeans(k=2, m=1.5)
fcm.fit(all_features)
fuzzy_membership_matrix = fcm.fuzzy_labels_
labels = np.argmax(fuzzy_membership_matrix, axis=1)


# Figure out which pieces of data are at what point, labelled by the indices.
groups = {}
for i in range(0, len(labels)):
    cluster = labels[i]
    if labels[i] not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(i)
    else:
        groups[cluster].append(i)

print(fuzzy_membership_matrix)

# View clusters

# Getting unique labels
u_labels = np.unique(labels)

from sklearn.metrics import accuracy_score, classification_report

# Assuming you have true labels


targets = test_dataset.targets
array_of_arrays = np.array([[int(char) for char in string] for string in targets])

# Transpose the array
true_lab = array_of_arrays.T

#For model 0
true_labels = true_lab[2]


# Map cluster labels to the most frequent true class in each cluster
# cluster_to_class = {}
# for cluster in np.unique(labels):
#     mask = (labels == cluster)
#     most_frequent_class = np.argmax(np.bincount(true_labels[mask]))
#     cluster_to_class[cluster] = most_frequent_class
#
# # Map cluster labels to predicted labels
# predicted_labels = np.array([cluster_to_class[cluster] for cluster in labels])

predicted_labels = labels

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

print(classification_report(true_labels, abs(predicted_labels-1), target_names=['Other', '[0,2]']))
