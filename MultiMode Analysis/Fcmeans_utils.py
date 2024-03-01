import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


class CombinedModel(nn.Module):
    def __init__(self, models):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        combined_output = torch.cat(outputs, dim=1)  # Concatenate features
        return combined_output

def view_cluster(cluster, groups, test_dataset):
    indices = groups[cluster]
    if len(indices) > 30:
        indices = indices[:29]

    plt.figure(figsize=[25, 25])

    for i in range(0, len(indices)):
        plt.subplot(10, 10, i + 1)
        img = test_dataset[indices[i]][0].cpu().numpy()
        img = img[0]
        plt.imshow(img)


import numpy as np


def custom_accuracy(array1, array2):
    diff = np.abs(np.where(array2 > 0.2, np.ceil(array2), 0) - array1)
    diff = np.sum(diff)
    print(1 - diff / 10000)
    return 1 - diff / 10000


def match_rows(array1, array2):

    num_rows1 = array1.shape[0]
    num_rows2 = array2.shape[0]

    # Initialize array to keep track of matched rows in array2
    matched_rows = np.zeros(num_rows2, dtype=bool)

    # Initialize array to store matched indices
    matched_indices = np.zeros(num_rows1, dtype=int)

    # Calculate accuracy between rows of both arrays
    for i in range(num_rows1):
        max_accuracy = -1
        max_index = -1
        for j in range(num_rows2):
            if not matched_rows[j]:
                accuracy = custom_accuracy(array1, array2[matched_indices])
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_index = j
        matched_indices[i] = max_index
        matched_rows[max_index] = True

    return matched_indices



