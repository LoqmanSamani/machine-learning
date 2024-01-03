import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs


# A model (neural network) with two layers (input layer, hidden layer(1), output layer) to solve a classification problem


class Classification(object):

    def __init__(self, epsilon=1e-2, sigma=1e-8):

        self.epsilon = epsilon
        self.sigma = sigma
        self.cost = []
        self.params = {}


    def layer_size(self, X, Y, H):

        X_n = X.shape[0]
        Y_n = Y.shape[0]
        H_n = H

        return X_n, Y_n, H_n


    def init_params(self, X_n, Y_n, H_n, epsilon):

        W1 = np.random.randn(H_n, X_n) * epsilon
        b1 = np.zeros((H_n, 1))
        W2 = np.random.randn(Y_n, H_n) * epsilon
        b2 = np.zeros((Y_n, 1))

        params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return params


    def sigmoid(self, Z):

        A = 1 / (1 + np.exp(-Z))

        return A


    def for_propagation(self, X, params):

        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]

        Z1 = np.matmul(W1, X) + b1
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }

        return cache


    def log_loss(self, Y, A2, sigma):

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A2 + sigma) + (1 - Y) * np.log(1 - A2 + sigma)) / m

        return cost


    def back_propagation(self, params, cache, X, Y):

        W2 = params["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return grads


    def update(self, params, grads, alpha):

        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 -= dW1 * alpha
        b1 -= db1 * alpha
        W2 -= dW2 * alpha
        b2 -= db2 * alpha

        params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return params


    def train(self, X, Y, H,alpha=1e-2, num_iters=100):

        X_n, Y_n, H_n = self.layer_size(X=X, Y=Y, H=H)
        self.params = self.init_params(X_n, Y_n, H_n, self.epsilon)

        for i in range(num_iters):
            """
            cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
            """

            cache = self.for_propagation(
                X=X,
                params=self.params
            )

            cost = self.log_loss(
                Y=Y,
                A2=cache["A2"],
                sigma=self.sigma
            )

            grads = self.back_propagation(
                params=self.params,
                cache=cache,
                X=X,
                Y=Y
            )

            params = self.update(
                params=self.params,
                grads=grads,
                alpha=alpha
            )

            self.cost.append(cost)
            self.params = params


    def predict(self, X):

        cache = self.for_propagation(
            X=X,
            params=self.params
        )

        A2 = cache["A2"]

        predicted = (A2 > 0.5).astype(int)

        return predicted


# Generate train data
np.random.seed(42)
m = 2000
samples, labels = make_blobs(n_samples=m,
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
                             cluster_std=1.1,
                             random_state=0)


Y = np.array([1 if label == 0 or label == 1 else 0 for label in labels]).reshape(1, len(labels))
X1 = np.transpose(samples)

print(Y.shape)
print(X1.shape)
"""
(1, 2000)
(2, 2000)
"""

plt.scatter(samples[:, 0], samples[:, 1], c=Y, cmap=colors.ListedColormap(['blue', 'green']))
plt.title("Generated Data for Classification")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


X_train = X1[:, :1950]
Y_train = Y[:, :1950]
X_test = X1[:, 1950:]
Y_test = Y[:, 1950:]

print(X_train.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X_test.shape)
"""
(2, 1950)
(1, 1950)
(1, 50)
(2, 50)
"""

model = Classification()
model.train(X=X_train, Y=Y_train, H=2, alpha=0.5, num_iters=3000)

print(model.params)
"""
{
 'W1': array([[ 1.3259593 , -1.15532004],
              [ 1.42244746, -1.25914247]]),
              
 'b1': array([[-3.31067647],
              [ 3.99611548]]), 
              
 'W2': array([[-8.5706773 ,  8.26227148]]), 
 
 'b2': array([[-3.84971523]])}
"""

plt.plot(model.cost)
plt.title("Model 4 Cost")
plt.xlabel("Iteration")
plt.ylabel("cost")
plt.show()


predicted = model.predict(X=X_test)

print(Y_test)
print(predicted)
"""
[[0 1 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 0 0 1 0 0 0 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 1 1 0 0 1 0 1 1 1 1]]
[[0 1 1 0 1 0 1 1 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 0 1 0 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1]]
"""

accuracy = np.mean(Y_test == predicted)
print(accuracy)
""" 0.92 """





x_min, x_max1 = X1[0, :].min() - 1, X1[0, :].max() + 1
y_min, y_max = X1[1, :].min() - 1, X1[1, :].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max1, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Flatten the mesh grid and predict the class for each point
grid_points = np.c_[xx.ravel(), yy.ravel()].T
Z = model.predict(grid_points)

# Reshape the predictions to the shape of the mesh grid
Z = Z.reshape(xx.shape)

# Plot the original data points
plt.scatter(samples[:, 0], samples[:, 1], c=Y, cmap=colors.ListedColormap(['blue', 'green']))

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=colors.ListedColormap(['lightblue', 'lightgreen']), alpha=0.3)

plt.title("Decision Boundary (Model 4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# Example 2
np.random.seed(3)
m1 = 3000
samples1, labels1 = make_blobs(n_samples=m1,
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
                             cluster_std=1.1,
                             random_state=0)


ls = [0, 1, 2]

Y1 = np.array([1 if label in ls else 0 for label in labels1]).reshape(1, len(labels1))
X1 = np.transpose(samples1)

print(Y1.shape)
print(X1.shape)
"""
(1, 3000)
(2, 3000)
"""

plt.scatter(samples1[:, 0], samples1[:, 1], c=Y1, cmap=colors.ListedColormap(['red', 'yellow']))
plt.title("Generated Data for Classification (model4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


X1_train = X1[:, :2900]
Y1_train = Y1[:, :2900]
X1_test = X1[:, 2900:]
Y1_test = Y1[:, 2900:]

print(X1_train.shape)
print(Y1_train.shape)
print(Y1_test.shape)
print(X1_test.shape)
"""
(2, 2900)
(1, 2900)
(1, 100)
(2, 100)
"""


model1 = Classification()
model1.train(X=X1_train, Y=Y1_train, H=2, alpha=0.6, num_iters=1000)

print(model1.params)
"""
{
 'W1': array([[-1.08567911,  0.94216258],
              [-1.08808215,  0.97699285]]), 
              
 'b1': array([[2.19310616],
              [2.12699213]]),
               
 'W2': array([[5.21726498, 5.11102064]]), 
 
 'b2': array([[-3.80087554]])
 }
"""

plt.plot(model1.cost)
plt.title("Model 4-1 Cost")
plt.xlabel("Iteration")
plt.ylabel("cost")
plt.show()

predicted1 = model1.predict(X=X1_test)


accuracy1 = np.mean(Y1_test == predicted1)
print(accuracy1)
""" 0.99 """



# Generate a mesh grid
x1_min, x1_max = X1[0, :].min() - 1, X1[0, :].max() + 1
y1_min, y1_max = X1[1, :].min() - 1, X1[1, :].max() + 1
xx1, yy1 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                       np.arange(y1_min, y1_max, 0.01))

# Flatten the mesh grid and predict the class for each point
grid_points1 = np.c_[xx1.ravel(), yy1.ravel()].T
Z1 = model1.predict(grid_points1)

# Reshape the predictions to the shape of the mesh grid
Z1 = Z1.reshape(xx1.shape)

# Plot the original data points
plt.scatter(samples1[:, 0], samples1[:, 1], c=Y1, cmap=colors.ListedColormap(['red', 'yellow']))

# Plot the decision boundary
plt.contourf(xx1, yy1, Z1, cmap=colors.ListedColormap(['lightcoral', 'lightyellow']), alpha=0.3)

plt.title("Decision Boundary (Model 4-1)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

