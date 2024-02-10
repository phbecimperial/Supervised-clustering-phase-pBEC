import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from tqdm import tqdm

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

# batch_size = 64
new_size = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("Saved_models/Model_3_ResNet", map_location=torch.device('cpu'))
#model = torch.load("CNN.pt", map_location=torch.device('cpu'))

newmodel = torch.nn.Sequential(*(list(model.children())[:-2])) #-1 for ResNet, -2 for CNN

#newmodel = CustomModel(model)


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (
#     0.5,))])  # Transforms it to a tensor, and rescales pixel values [0, 1]
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


test_dataset = CustomDataset(root_dir='Training_images', new_size=new_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False) #Get images one at a time

with torch.no_grad():
    model.eval()

    all_features = []

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)

        outputs = newmodel(inputs)
        outputs = outputs.cpu().numpy()
        outputs = outputs[0].tolist()
        all_features.append(outputs)

all_features = np.array(all_features)

#Insert PCA if needed
from sklearn.decomposition import PCA
all_features = all_features.reshape(1000, -1)

all_features = PCA(512).fit_transform(all_features)

target_names = test_dataset.classes
kmeans = KMeans(n_clusters=len(target_names), verbose=1)
kmeans.fit(all_features)

# Figure out which pieces of data are at what point, labelled by the indices.
groups = {}
for i in range(0, len(kmeans.labels_)):
    cluster = kmeans.labels_[i]
    if kmeans.labels_[i] not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(i)
    else:
        groups[cluster].append(i)


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


view_cluster(3)
plt.show()

# Getting unique labels
u_labels = np.unique(kmeans.labels_)

# plotting the results:

# for i in u_labels:
#     plt.scatter(all_features[kmeans.labels_ == i, 0], all_features[kmeans.labels_ == i, 1], label=i)
# plt.legend()
# plt.show()


from sklearn.metrics import accuracy_score, classification_report

# Assuming you have true labels for the MNIST dataset
true_labels = test_dataset.targets.numpy()

# Map cluster labels to the most frequent true class in each cluster
cluster_to_class = {}
for cluster in np.unique(kmeans.labels_):
    mask = (kmeans.labels_ == cluster)
    most_frequent_class = np.argmax(np.bincount(true_labels[mask]))
    cluster_to_class[cluster] = most_frequent_class

# Map cluster labels to predicted labels
predicted_labels = np.array([cluster_to_class[cluster] for cluster in kmeans.labels_])

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

print(classification_report(true_labels, predicted_labels, target_names=test_dataset.classes))
