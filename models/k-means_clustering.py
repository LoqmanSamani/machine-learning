import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeansClustering(object):
    def __init__(self, init_centroids=None, epochs=10):

        if init_centroids:
            self.centroids = init_centroids
        else:
            self.centroids = None
        self.epochs = epochs

        self.cost = []
        self.centroids = None
        self.indices = None

        self.colors_named = ['blue', 'green', 'red', 'cyan', 'magenta',
                                 'yellow', 'black', 'white', 'orange', 'purple',
                                 'brown', 'pink', 'gray', 'olive', 'teal', 'navy',
                                 'maroon', 'turquoise', 'indigo', 'salmon']

    def centroid_init(self, X, K):

        indices = np.random.permutation(X.shape[0])
        centroids = X[indices[:K]]

        return centroids

    def compute_distortion(self, X, centroids, indices):

        distortion = 0
        for i, centroid in enumerate(centroids):
            distortion += np.sum(np.power(X[indices == i] - centroid, 2))

        return distortion

    def closest_centroids(self, X, centroids):

        K = centroids.shape[0]
        indices = np.zeros(X.shape[0], dtype=int)

        for i, x in enumerate(X):

            dist = np.zeros(K)
            for j, centroid in enumerate(centroids):
                dist[j] = np.sum(np.power(x - centroid, 2))
            indices[i] = np.argmin(dist)

        return indices

    def compute_centroids(self, X, indices, K):

        centroids = np.zeros((K, X.shape[1]))
        for i in range(K):

            centroid = np.mean(X[indices == i], axis=0)
            centroids[i, :] = centroid

        return centroids
    
    
    def predict(self, X_new):
    
        indices = []

        if len(X_new) == 0:
            raise ValueError("New example is empty !!!")
        if X_new.shape[1] != self.centroids.shape[1]:
            raise ValueError("New example and centroids must have the same number of columns")
    
        if self.centroids is None:
            raise ValueError("Fit method must be called first to initialize centroids.")
        
        elif len(X_new) == 1:
            K = self.centroids.shape[0]
            dist = np.zeros(K)
            for i, centroid in enumerate(self.centroids):
                dist[i] = np.sum(np.power(X_new - centroid, 2))
            indices.append(np.argmin(dist))
        else:
            ind = self.closest_centroids(X_new, self.centroids)
            indices = list(ind)
                              
        return indices
    
    
    def plot(self, X, centroids, indices, colors=None):

        if not colors:
            colors = self.colors_named
        if len(colors) < centroids.shape[0]:
            raise ValueError("The number of colors in colors must be equal or more than the number of centroids.")
        classes = {}
        for i in range(centroids.shape[0]):

            val = []
            for j, index in enumerate(indices):
                if index == i:
                    val.append(X[j])
            classes[f"C{i}"] = np.array(val)

        for n, val in enumerate(classes.values()):

            plt.scatter(val[:, 0], val[:, 1], color=colors[n])
            n += 1
        plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X")
        plt.title("Clustering with K-Means Algorithm")
        plt.show()


    

    def fit(self, X, K):

        centroids = self.centroid_init(X, K)

        for i in range(self.epochs):
            print(f"K-Means Iteration: {i + 1}")

            indices = self.closest_centroids(
                X=X,
                centroids=centroids
            )
            loss = self.compute_distortion(
                X=X,
                centroids=centroids,
                indices=indices
            )
            self.cost.append(loss)
            centroids = self.compute_centroids(
                X=X,
                indices=indices,
                K=K
            )
            self.indices = indices
            self.centroids = centroids


data = make_blobs(n_samples=5000,
                   n_features=2,
                   centers=15,
                   cluster_std=1.0,
                   center_box=(-10.0, 10.0),
                   shuffle=True,
                   random_state=3,
                   return_centers=True
                   )


print(data[0])
"""
[[-2.28376368  0.11281759]
 [-8.44091463  1.55130439]
 [-9.1063967   1.96290509]
 ...
 [-9.58034548 -2.39627635]
 [-5.01274278 -5.61954262]
 [-8.54129714 -6.14164407]]
 """

plt.scatter(data[0][:, 0], data[0][:, 1])
plt.show()


model = KMeansClustering()
model.fit(data[0], 15)
"""
K-Means Iteration: 1
K-Means Iteration: 2
K-Means Iteration: 3
K-Means Iteration: 4
K-Means Iteration: 5
K-Means Iteration: 6
K-Means Iteration: 7
K-Means Iteration: 8
K-Means Iteration: 9
K-Means Iteration: 10
"""
print(model.centroids)
"""
[[ 0.67735249  5.80781786]
 [-2.85965869  3.84068934]
 [-9.56913978  0.63241003]
 [ 2.56826549  2.69360441]
 [-1.82998128 -6.67158229]
 [ 7.58420411  8.6183268 ]
 [-5.91875305 -0.94820947]
 [-4.82975295  4.14831858]
 [-2.37330184  8.68817392]
 [ 8.05726489  7.01686625]
 [-4.22576229 -2.75062785]
 [-9.12136575 -1.80325008]
 [ 2.94861909 -4.48708399]
 [-6.83418571 -5.91248037]
 [-4.15443609  0.58725293]]
"""
print(model.indices)
"""
[14  2  2 ... 11 13 13]
"""
print(model.cost)
"""
[30550.585222169328, 21944.292916764247, 17641.884322105874, 15845.09120186005, 
15020.153623414919, 13999.067727209493, 13337.357461613312, 12744.342954703863, 
11871.720539215989, 10881.03276598686]
"""

model.plot(X=data[0], centroids=model.centroids, indices=model.indices)

