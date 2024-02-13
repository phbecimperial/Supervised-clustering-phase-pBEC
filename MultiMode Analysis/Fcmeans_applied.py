import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from tqdm import tqdm
from pickle_Dataset import CustomDataset
from Fcmeans_utils import CustomModel
import torch

all_pred = []
for i in range(0, 13):
    phase = i

    new_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Models/Res_Class_{}.pt".format(phase))
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
    target_names = test_dataset.classes

    #%%
    #print("INFO: Starting clustering")
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

    #print(fuzzy_membership_matrix)
    # Getting unique labels
    u_labels = np.unique(labels)

    from sklearn.metrics import accuracy_score, classification_report

    # Assuming you have true labels
    targets = test_dataset.targets
    array_of_arrays = np.array([[int(char) for char in string] for string in targets])
    true_lab = array_of_arrays.T

    #Depending on the model
    true_labels = true_lab[phase]


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
    print('Labels and counts', np.unique(predicted_labels, return_counts=True))

    print(classification_report(true_labels, predicted_labels, target_names=['A', 'B']))

    all_pred.append(predicted_labels)


diff = np.abs(all_pred - true_lab)
diff = np.sum(diff)
print(1 - diff/10000)

