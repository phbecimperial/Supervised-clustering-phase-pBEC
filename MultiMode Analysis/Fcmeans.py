import matplotlib.pyplot as plt
import torch
import numpy as np
# from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fcmeans import FCM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("cnn.pt", map_location=torch.device('cpu'))
newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (
    0.5,))])  # Transforms it to a tensor, and rescales pixel values [0, 1]
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    model.eval()

    all_features = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = newmodel(inputs)
        outputs = outputs.cpu().numpy()
        outputs = outputs[0].tolist()
        all_features.append(outputs)

all_features = np.array(all_features)

# Insert PCA if needed

from sklearn.decomposition import PCA

# all_features = PCA(2).fit_transform(all_features)

target_names = test_dataset.classes
fcm = FCM(n_clusters=len(target_names), m=2)
fcm.fit(all_features)

labels = fcm.predict(all_features)
# Figure out which pieces of data are at what point, labelled by the indices.
groups = {}
for i in range(0, len(labels)):
    cluster = labels[i]
    if labels[i] not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(i)
    else:
        groups[cluster].append(i)

#percentages = fcm.soft_predict(all_features)

# %%
# View clusters
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


view_cluster(2)
plt.show()

# Getting unique labels
u_labels = np.unique(labels)

# plotting the results:

# for i in u_labels:
#     plt.scatter(all_features[labels == i, 0], all_features[labels == i, 1], label=i)
# plt.legend()
# plt.show()

from sklearn.metrics import accuracy_score, classification_report

# Assuming you have true labels for the MNIST dataset
true_labels = test_dataset.targets.numpy()

# Map cluster labels to the most frequent true class in each cluster
cluster_to_class = {}
for cluster in np.unique(labels):
    mask = (labels == cluster)
    most_frequent_class = np.argmax(np.bincount(true_labels[mask]))
    cluster_to_class[cluster] = most_frequent_class

# Map cluster labels to predicted labels
predicted_labels = np.array([cluster_to_class[cluster] for cluster in labels])

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

print(classification_report(true_labels, predicted_labels, target_names=test_dataset.classes))
