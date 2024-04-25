import pickle
import numpy as np
import torch
from os.path import sep
from Fcmeans_utils import CustomModel, CombinedModel
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pickle_Dataset import CustomDataset, Predict_Dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from glob import glob
from torchvision import transforms
from PIL import Image
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
import cv2
from scipy.ndimage import zoom

def sigmoid(x, k, c):
    return 1/(1+np.exp(-(x - c) / k))

# Load centroids and membership values from file

def predict(data, cluster_centres, m, classifer_model):
        D = 1.0 / euclidean_distances(data, cluster_centres, squared=True)
        D **= 1.0 / (m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        return D


def Kmeans_no_CNN(files, num_clusters, powers, lengths):

    img_features = []
    for i, f in enumerate(files):
        image = cv2.imread(f, 0)
        
        image = zoom(image, (224/image.shape[0], 244/image.shape[1]))

        image = image.flatten()

        
        # if powers[i] is not None:
        #     image = np.concatenate((image, [powers[i]]))
        
        # if lengths[i] is not None:
        #     image = np.concatenate((image, [lengths[i]]))
        
        img_features.append(image)

    img_features = np.array(img_features)

    

    pca = PCA(n_components=100, random_state=22)

    pca.fit(img_features)

    x = pca.transform(img_features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    kmeans.fit(x)

    return kmeans.labels_, None


def predict_full_output(files, phases, root, sigm = False):

    print(files)
    mod_files = glob(root + r'*.pt')
    sort_args = np.argsort([i.split(sep)[-1].split('_')[-1][:-3] for i in mod_files])
    mod_files = np.array(mod_files)[sort_args]


    img_features = []
    preds = []
    with torch.no_grad():

        for i, f in tqdm(enumerate(files)):

            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

            all_features = []
            ind_preds = []
            for i in range(phases):

                model = torch.load(mod_files[i])
                transform = v2.Compose([v2.ToTensor(), 
                                        v2.Resize((224,224), antialias=True), 
                                        v2.Normalize((0.5,), (0.5,))])

                # Load the image
                #image_path = r'C:\Data\Phase\pbecCropc_20240222_210313_42951.0_0.08428571428571428_950.8663940429688_.bmp'
                if sigm:
                    image = cv2.imread(f,0)
                    image = (image - np.min(image))/(np.max(image) - np.min(image))
                    image = sigmoid(image, 0.05, 0.1)
                    image = np.array(image, np.float32)

                else:
                    image = Image.open(f)
                
                # Apply transformations
                input_image = transform(image)
                input_image = input_image.unsqueeze(0)

                model.eval()

                inputs = input_image.to(device)
                outputs = model(inputs)
                #outputs = outputs.cpu().numpy()
                outputs = torch.softmax(outputs,1)
                outputs = outputs[0].cpu().numpy()

                all_features.append(outputs)
                ind_preds.append(np.argmin(outputs))



            img_features.append(np.array(all_features))
            preds.append(ind_preds)
    x = np.array(img_features)
    preds = np.array(preds)
    return x, preds


def predeict_images_CNN(files, root, phases = 10,powers = None, lengths = None, sigm = False):

    print(files)

    img_features = []
    img_pl_features = []
    mod_files = glob(root + r'*.pt')

    sort_args = np.argsort([i.split(sep)[-1].split('_')[-1][:-3] for i in mod_files])
    mod_files = np.array(mod_files)[sort_args]
    with torch.no_grad():

        for i, f in tqdm(enumerate(files)):

            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

            pl_features = []
            all_feats = []
            for i in range(phases):
                model = torch.load(mod_files[i])
                newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
                newmodel = CustomModel(model)

                transform = v2.Compose([v2.ToTensor(), 
                                        v2.Resize((224,224), antialias=True), 
                                        v2.Normalize((0.5,), (0.5,))])

                # Load the image
                #image_path = r'C:\Data\Phase\pbecCropc_20240222_210313_42951.0_0.08428571428571428_950.8663940429688_.bmp'
                if sigm:
                    image = cv2.imread(f,0)
                    image = (image - np.min(image))/(np.max(image) - np.min(image))
                    image = sigmoid(image, 0.05, 0.1)
                    image = np.array(image, np.float32)


                else:
                    image = Image.open(f)
                # Apply transformations
                input_image = transform(image)
                input_image = input_image.unsqueeze(0)

                model.eval()

                inputs = input_image.to(device)
                outputs = newmodel(inputs)
                outputs = outputs.cpu().numpy()

                outputs = outputs[0]

                if (powers is not None) and (lengths is not None):


                    ploutputs = np.concatenate((outputs, [powers[i]]))
                
                    ploutputs = np.concatenate((outputs, [lengths[i]]))

                    pl_features.append(ploutputs)

                all_feats.append(outputs)



            img_features.append(np.array(all_feats).flatten())
            if powers is not None:
                img_pl_features.append(np.array(pl_features).flatten())


    pca = PCA(n_components=100, random_state=22)
    pca.fit(img_features)
    x = pca.transform(img_features)
    x = np.array(x)

    if powers is not None:
        pca1 = PCA(n_components=100, random_state=22)
        pca1.fit(img_pl_features)
        y = pca1.transform(img_pl_features)
        y = np.array(y)

        return x, y


    return x 
  

    # pca = PCA(n_components=100, random_state=22)

    # pca.fit(img_features)

    # x = pca.transform(img_features)


def quick_kmeans(features, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    kmeans.fit(features)


    return kmeans.labels_, None


def quick_fcmeans(features, num_clusters, m = 1.5, return_matrix = False):

    fcm = FuzzyKMeans(k=num_clusters, m=m)
    fcm.fit(features)
    fuzzy_membership_matrix = fcm.fuzzy_labels_
    fuzzy_membership_matrix = fuzzy_membership_matrix.T


    if return_matrix:
        return fuzzy_membership_matrix 

    alpha = np.max(fuzzy_membership_matrix, axis=0)
    labels = np.argmax(fuzzy_membership_matrix, axis=0)

    return labels, alpha



def predeict_images_fc(files, phases=10, num_clusters=10, m=1.5):
    print(files)

    img_features = []

    with torch.no_grad():

        models = []
        for i in range(phases):
            model = torch.load("MultiMode Analysis/Models/Res_Class_{}.pt".format(i))
            newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
            newmodel = CustomModel(model)
            models.append(newmodel)

        for i, f in tqdm(enumerate(files)):

            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

            all_features = []
            newmodel = CombinedModel(models)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to a fixed size
                transforms.ToTensor()  # Convert the image to a PyTorch tensor
            ])

            # Load the image
            image = Image.open(f)
            # Apply transformations
            input_image = transform(image)
            input_image = input_image.unsqueeze(0)

            model.eval()

            inputs = input_image.to(device)
            outputs = newmodel(inputs)
            outputs = outputs.cpu().numpy()
            outputs = outputs[0]
            img_features.append(outputs)

    img_features = np.array(img_features)

    pca = PCA(n_components=100, random_state=22)

    pca.fit(img_features)

    x = pca.transform(img_features)

    fcm = FuzzyKMeans(k=num_clusters, m=1.5)
    fcm.fit(x)
    fuzzy_membership_matrix = fcm.fuzzy_labels_
    fuzzy_membership_matrix = fuzzy_membership_matrix.T

    alpha = np.max(fuzzy_membership_matrix, axis=0)
    labels = np.argmax(fuzzy_membership_matrix, axis=0)

    return labels, alpha


           

           


#Get the new data

root_dir = 'INSERT HERE'

if __name__ == '__main__':
    #files = glob(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\MultiMode Analysis\20240222\Cropped images\*.bmp')
    with open(r'MultiMode Analysis\stim_files.pkl', 'rb') as f:
        files = pickle.load(f)
    int_times = []
    powers = []
    lengths = []
    for file in files:
        split_file = file.split('_')
        int_times.append(float(split_file[3]))
        powers.append(float(split_file[4]))
        lengths.append(float(split_file[5]))

    int_times =  np.array(int_times)
    files = np.array(files)
    powers = np.array(powers)
    lengths = np.array(lengths)

    model_root = r'C:\Users\Pouis\Documents\Uni Shit\Masters\PhaseGit\Supervised-clustering-phase-pBEC\MultiMode Analysis\Models\Apr2402'

    features, plfeatures = predeict_images_CNN(files,model_root,5,powers, lengths,True)

    with open('Apr_2402_POWLEN_features.pkl', 'wb') as f:
        pickle.dump(plfeatures, f)

    with open('Apr_2402_features.pkl', 'wb') as f:
        pickle.dump(features, f)
 
    out, preds = predict_full_output(files, 5, model_root, True)

    with open('Apr_2402_CNN_out.pkl', 'wb') as f:
        pickle.dump((out, preds), f)


    #labels, _  = Kmeans_no_CNN(files, 13, powers, lengths)

    with open('Apr_2402_features.pkl', 'rb') as f:
        features = pickle.load(f)

    # labels, alphas = quick_fcmeans(features, num_clusters= 6, m= 1.5)

    # print(labels, alphas)

    for i in range(6,14):
        
        # labels1, _ = quick_kmeans(features1, i)
        labels2, _ = quick_kmeans(features, i)
        #labels, _ = Kmeans_no_CNN(files,9,powers,lengths)

        # with open(f'Apr_20_POWLEN_predicted_labels_{i}.pkl', 'wb') as f:
        #     pickle.dump((labels1), f)

        with open(f'Apr_2402_predicted_labels_{i}.pkl', 'wb') as f:
            pickle.dump((labels2), f)


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

   