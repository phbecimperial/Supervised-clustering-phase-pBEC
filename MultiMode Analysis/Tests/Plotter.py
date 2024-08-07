import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
#from fcmeans import FCM
from tqdm import tqdm
from pickle_Dataset import CustomDataset

new_size = (224, 224)
test_dataset = CustomDataset(root_dir='Training_images', new_size=(224, 224))
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# Assuming you have true labels
targets = test_dataset.targets
array_of_arrays = np.array([[int(char) for char in string] for string in targets])
true_lab = np.abs(array_of_arrays-1)

from sklearn.decomposition import PCA

# Initialise
pca = PCA(n_components=2)
pca.fit(true_lab)

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Principal components
print("Principal components:", pca.components_)

result = pca.transform(true_lab)

# Assuming X_pca has shape (n_samples, 2)
plt.scatter(result[:, 0], result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()



