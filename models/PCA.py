import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

"""  Implement Principal Component Analysis(PCA) with numpy"""


class PrincipalComponentAnalysis(object):
    def __init__(self, num_components):
        self.num_components = num_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        data = data - self.mean

        covariance = np.cov(data.T)

        eigenvectors, eigenvalues = np.linalg.eig(covariance)
        eigenvectors = eigenvectors.T

        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        self.components = eigenvectors[:self.num_components]
        self.explained_variance = eigenvalues[:self.num_components] / np.sum(eigenvalues)

    def transform(self, data):
        data = data - self.mean
        return np.dot(data, self.components.T)

    def inverse_transform(self, data):
        return np.dot(data, self.components) + self.mean

    def get_components(self):
        return self.components

    def get_mean(self):
        return self.mean


data, target = make_blobs(n_samples=1000, n_features=10, random_state=2)
print(data.shape)

pca_sklearn = PCA(n_components=2)
pca_sklearn.fit(data)
pca_data = pca_sklearn.transform(data)
print(pca_data.shape)
print("scikit-learn PCA", pca_sklearn.explained_variance_ratio_)

pca = PrincipalComponentAnalysis(num_components=2)
pca.fit(data=data)
my_pca_data = pca.transform(data)
print(my_pca_data.shape)
print("my PCA explained variance", pca.explained_variance)

