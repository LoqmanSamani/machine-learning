import matplotlib.pyplot as plt
import numpy as np
import time


""" Neural Networks for Handwritten Digit Recognition, Multiclass """


class NNL2(object):

    def __init__(
            self, epochs=50, params=None, alpha=1e-2, beta=1e-2, sigma=1e-8, lam=1, print_cost=10, seed=42, batch=10
    ):

        self.epochs = epochs
        self.params = params
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.lam = lam
        self.print_cost = print_cost
        self.seed = seed
        self.batch = batch

        self.cost = np.zeros(epochs)
        self.time_ = {}

    def layer_size(self, X_train, Y_train):

        layer_in = X_train.shape[0]
        layer_out = len(np.unique(Y_train))

        return layer_in, layer_out

    def param_init(self, layer_in, layer_out, units, beta):

        params = {}

        for i in range(len(units)):
            if i == 0:
                params[f"W{i+1}"] = np.random.randn(units[i], layer_in) * beta
                params[f"b{i+1}"] = np.random.randn(units[i], 1) * beta
            elif i == len(units)-1:
                params[f"W{i+1}"] = np.random.randn(layer_out, units[i-1]) * beta
                params[f"b{i+1}"] = np.random.randn(layer_out, 1) * beta
            else:
                params[f"W{i+1}"] = np.random.randn(units[i], units[i-1]) * beta
                params[f"b{i+1}"] = np.random.randn(units[i], 1) * beta

        return params

    def softmax(self, Z):

        A = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A /= np.sum(A, axis=1, keepdims=True)
        return A

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def cross_entropy(self, Y, A):

        A = np.clip(A, 1e-15, 1 - 1e-15)

        loss = -np.mean(np.sum(Y * np.log(A), axis=1))

        return loss

    def forward_propagation(self, X, W, b):

        Z = np.matmul(W, X) + b

        return Z

    def back_propagation(self, X, Y, A, W, lam):

        m = X.shape[1]
        dA = A - Y
        dW = (np.matmul(dA, X.T) + (lam * W)) / m
        db = np.sum(dA, axis=1, keepdims=True) / m

        return dW, db

    def param_update(self, W, b, dW, db, alpha):

        W -= alpha * dW
        b -= alpha * db

        return W, b

    def train(self, X_train, Y_train, units):
        """
        X_train.shape = (number of features, number of instances)
        Y_train.shape = (1, number of instances)
        units = a list which contains the number of units in each layer
        """
        layer_in, layer_out = self.layer_size(
            X_train=X_train,
            Y_train=Y_train
        )

        np.random.seed(self.seed)

        if not self.params:
            self.params = self.param_init(
                layer_in=layer_in,
                layer_out=layer_out,
                units=units,
                beta=self.beta
            )


        for i in range(self.epochs):

            tick = time.time()

            indices = [np.random.choice(range(X_train.shape[1])) for _ in range(self.batch)]
            X_batch = np.array([X_train[:, x] for x in indices]).T
            Y_batch = np.array([Y_train[:, y] for y in indices]).T

            A = X_batch
            for j in range(1, len(units)+1):

                Z = self.forward_propagation(
                    X=X_batch,
                    W=self.params[f"W{j}"],
                    b=self.params[f"b{j}"]
                )

                if j != len(units):
                    A = self.relu(Z=Z)
                else:
                    A = self.softmax(Z=Z)

            cost = self.cross_entropy(Y=Y_batch, A=A)

            grads = {}
            iters = len(units)

            while iters > 0:

                dW, db = self.back_propagation(
                    X=X_batch,
                    Y=Y_batch,
                    A=A,
                    W=self.params[f"W{iters}"],
                    lam=self.lam
                )

                grads[f"dW{iters}"] = dW
                grads[f"db{iters}"] = db
                iters -= 1

            up = len(units)
            params = {}
            while up > 0:
                W, b = self.param_update(
                    W=self.params[f"W{up}"],
                    b=self.params[f"b{up}"],
                    dW=grads[f"dW{up}"],
                    db=grads[f"db{up}"],
                    alpha=self.alpha
                )

                params[f"W{up}"] = W
                params[f"b{up}"] = b

            self.params = params
            self.cost[i] = cost

            tock = time.time()

            if i % self.print_cost == 0:
                print(f"Epoch: {i}, Cost: {cost}, Time: {tock - tick}")





X = np.load("/home/sam/projects/machine-learning/data/handwritten/X.npy")
print(X.shape)
""" (5000, 400) """
print(X[0, 100:110])
""" 
[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  1.28335523e-17 -3.26286765e-04 -1.38651604e-02
  8.15651552e-02  3.82800381e-01]
"""

Y = np.load("/home/sam/projects/machine-learning/data/handwritten/y.npy")
print(Y.shape)
""" (5000, 1) """
print(Y[100:120, :])
"""
[[0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]]
"""






