import numpy as np
import time


""" Neural Networks for Handwritten Digit Recognition, Multiclass classification"""


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

    def forward_propagation(self, X, params, activations):

        cache = {}
        X=X
        for i in range(1, int((len(params.keys())/2) + 1)):
            Z = np.matmul(params[f"W{i}"], X)
            Z += params[f"b{i}"]
            cache[f"Z{i}"] = Z
            X = Z
            if activations[i-1] == "relu":
                A = self.relu(Z)
            else:
                A = self.softmax(Z)
            cache[f"A{i}"] = A

        return cache

    def back_propagation(self, X, Y, params, cache, lam):

        m = Y.shape[1]

        grads = {}
        st = int(len(params.keys()) / 2)
        while st > 0:
            if st == int(len(params.keys()) / 2):
                dZ = cache[f"A{st}"] - Y
            else:
                dZ = np.matmul(params[f"W{st}"].T, grads[f"dZ{st}"]) * cache[f"A{st-1}"] * (1 - cache[f"A{st-1}"])

            grads[f"dZ{st}"] = dZ

            if st == 1:
                dW = (np.matmul(grads["dZ1"], X.T)) + (lam * params[f"W{st}"]) / m
                db = np.sum(grads["dZ1"], axis=1, keepdims=True)
            else:
                dW = (np.matmul(grads[f"dZ{st}"], cache[f"A{st-1}"].T)) + (lam * params[f"W{st}"]) / m
                db = np.sum(grads[f"dZ{st}"], axis=1, keepdims=True)
            grads[f"dW{st}"] = dW
            grads[f"db{st}"] = db

        return grads

    def param_update(self, params, grads, alpha):

        for i in range(1, int((len(params.keys())/2) + 1)):

            params[f"W{i}"] -= grads[f"dW{i}"] * alpha
            params[f"b{i}"] -= grads[f"db{i}"] * alpha

        return params

    def train(self, X_train, Y_train, units, activations):
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

            cache = self.forward_propagation(
                X=X_batch,
                params=self.params,
                activations=activations
            )

            cost = self.cross_entropy(
                Y=Y_batch,
                A=cache[f"A{len(units)-1}"]
            )

            grads = self.back_propagation(
                X=X_batch,
                Y=Y_batch,
                params=self.params,
                cache=cache,
                lam=self.lam
            )

            params = self.param_update(
                params=self.params,
                grads=grads,
                alpha=self.alpha
            )

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

X_train = X[:3000, :]
Y_train = Y[:3000, :]
print(X_train.shape)
print(Y_train.shape)
"""
(3000, 400)
(3000, 1)
"""

X_test = X[3000:, :]
Y_test = Y[3000:, :]
print(X_test.shape)
print(Y_test.shape)
"""
(2000, 400)
(2000, 1)
"""

Xt= X_train.T
Yt = Y_train.T
print(X_train.shape)
print(Y_train.shape)
"""
(400, 3000)
(1, 3000)
"""


model = NNL2(epochs=50, params=None, alpha=0.2, beta=1e-2, sigma=1e-8, lam=1, print_cost=5, seed=42, batch=10)
model.train(X_train=Xt, Y_train=Yt, units=[25, 15, 10], activations=["relu", "relu", "softmax"])
