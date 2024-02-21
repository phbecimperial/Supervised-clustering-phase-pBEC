import pickle
import numpy as np
import torch
from Fcmeans_utils import CustomModel
from torch.utils.data import DataLoader
from pickle_Dataset import CustomDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances



# Load centroids and membership values from file

def predict(data, cluster_centres, m):
        D = 1.0 / euclidean_distances(data, cluster_centres, squared=True)
        D **= 1.0 / (m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        return D





#Get the new data

root_dir = 'INSERT HERE'

for i in range(0, 10):

    phase = i

    new_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Models/Res_Class_{}.pt".format(phase))
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    newmodel = CustomModel(model)

    #INSERT NEW DATA LOCATION HERE!
    test_dataset = CustomDataset(root_dir=root_dir, new_size=(224, 224))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        model.eval()

        all_features = []

        for inputs, labels in tqdm(test_loader, leave=True):


            inputs = inputs.to(device)

            outputs = newmodel(inputs)
            outputs = outputs.cpu().numpy()
            outputs = outputs[0].tolist()
            all_features.append(outputs) #Test

    all_features = np.array(all_features)

    with open('fuzzy_kmeans_clusters.pkl' + str(i), 'rb') as f:
        cluster_centres, fuzzmatrix, cluster_to_class = pickle.load(f)

    print(cluster_to_class)





    # Predict the clusters of new data points
    predicted_clusters = predict(all_features, cluster_centres)
    print("Predicted clusters for new data points:")
    print(predicted_clusters)

