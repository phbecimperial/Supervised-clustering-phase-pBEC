import pickle
import numpy as np
import torch
from Fcmeans_utils import CustomModel
from torch.utils.data import DataLoader
from pickle_Dataset import CustomDataset, Predict_Dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from glob import glob
from torchvision import transforms
from PIL import Image



# Load centroids and membership values from file

def predict(data, cluster_centres, m):
        D = 1.0 / euclidean_distances(data, cluster_centres, squared=True)
        D **= 1.0 / (m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        return D



def predeict_images(files, phases = 10, num_clusters = 10):

    print(files)

    img_features = []

    with torch.no_grad():

        for i, f in enumerate(files):

            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

            all_features = []

            for i in range(phases):
                model = torch.load("Models/Res_Class_{}.pt".format(i))
                newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
                newmodel = CustomModel(model)

                #test_dataset = Predict_Dataset([f])
                #loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

                # Define transformations to apply to the image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize the image to a fixed size
                    transforms.ToTensor()           # Convert the image to a PyTorch tensor
                ])

                # Load the image
                #image_path = r'C:\Data\Phase\pbecCropc_20240222_210313_42951.0_0.08428571428571428_950.8663940429688_.bmp'
                image = Image.open(f)
                # Apply transformations
                input_image = transform(image)
                input_image = input_image.unsqueeze(0)

                model.eval()

                inputs = input_image.to(device)
                outputs = newmodel(inputs)
                outputs = outputs.cpu().numpy()
                outputs = outputs[0]
                all_features.append(outputs)


            img_features.append(np.array(all_features).flatten())

    img_features = np.array(img_features)
    

    pca = PCA(n_components=100, random_state=22)

    pca.fit(img_features)

    x = pca.transform(img_features)


    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    kmeans.fit(x)


    return kmeans.labels_


    # groups = {}
    # for file, cluster in zip(files,kmeans.labels_):
    #     if cluster not in groups.keys():
    #         groups[cluster] = []
    #         groups[cluster].append(file)
    #     else:
    #         groups[cluster].append(file)
    



            

            


#Get the new data

root_dir = 'INSERT HERE'

if __name__ == '__main__':
    files = glob(r'C:\Data\Phase\*.bmp')
    #files = glob(r'Training_images_2')

    print(predeict_images(files,10,3))


    # for i in range(0, 10):

    #     phase = i

    #     new_size = (224, 224)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = torch.load("Models/Res_Class_{}.pt".format(phase))
    #     newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    #     newmodel = CustomModel(model)

    #     #INSERT NEW DATA LOCATION HERE!
    #     test_dataset = CustomDataset(root_dir=root_dir, new_size=(224, 224))
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    #     with torch.no_grad():
    #         model.eval()

    #         all_features = []

    #         for inputs, labels in tqdm(test_loader, leave=True):


    #             inputs = inputs.to(device)

    #             outputs = newmodel(inputs)
    #             outputs = outputs.cpu().numpy()
    #             outputs = outputs[0].tolist()
    #             all_features.append(outputs) #Test

    #     all_features = np.array(all_features)

    #     with open('fuzzy_kmeans_clusters.pkl' + str(i), 'rb') as f:
    #         cluster_centres, fuzzmatrix, cluster_to_class = pickle.load(f)

    #     print(cluster_to_class)





    #     # Predict the clusters of new data points
    #     predicted_clusters = predict(all_features, cluster_centres, 5)
    #     print("Predicted clusters for new data points:")
    #     print(predicted_clusters)

