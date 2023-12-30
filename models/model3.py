import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs

# a single perceptron neural network model to solve a simple classification problem

class Classification(object):

    def __init__(self):

        self.cost = []
        self.params = {}

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))

        return A

    def layer_size(self, X, Y):
        in_l, out_l = X.shape[0], Y.shape[0]

        return in_l, out_l

    def init_params(self, in_l, out_l):
        W = np.random.randn(out_l, in_l) * 0.01
        b = np.zeros((out_l, 1))

        params = {"W": W, "b": b}

        return params

    def for_prop(self, X, W, b):
        Z = np.matmul(W, X) + b

        return Z

    def log_loss(self, A, Y):

        log = - np.multiply(np.log(A), Y) - np.multiply(np.log(1 - A), 1 - Y)
        loss = np.mean(log)

        return loss

    def back_prop(self, A, Y, X):
        dZ = A - Y
        dW = 1 / X.shape[1] * np.dot(dZ, X.T)
        db = 1 / X.shape[1] * np.sum(dZ, axis=1, keepdims=True)

        grads = {"dW": dW, "db": db}

        return grads

    def update(self, W, b, dW, db, alpha):
        W -= dW * alpha
        b -= db * alpha

        return W, b

    def train(self, X, Y, num_iter=30, learning_rate=0.1):

        in_l, out_l = self.layer_size(X, Y)
        self.params = self.init_params(in_l, out_l)

        for i in range(num_iter):

            Z = self.for_prop(X, self.params["W"], self.params["b"])
            A = self.sigmoid(Z)
            grads = self.back_prop(A, Y, X)
            W, b = self.update(self.params["W"], self.params["b"], grads["dW"], grads["db"], learning_rate)
            self.cost.append(self.log_loss(A, Y))
            self.params = {"W": W, "b": b}

    def predict(self, X):

        Z = self.for_prop(X, self.params["W"], self.params["b"])
        A = self.sigmoid(Z)

        predict = A > 0.5

        return predict


#
X, Y = make_blobs(n_samples=1000,
                  centers=([2.5, 3], [6.7, 7.9]),
                  cluster_std=1.4,
                  random_state=0
                  )
print(X[:20])
print(Y[:20])
"""
[[ 1.03202585  1.01197489]
 [ 4.96967328  3.56022009]
 [ 6.00588631  9.6032088 ]
 [ 5.46350743  0.74477024]
 [ 5.67677698  5.60161548]
 [ 7.98997176  8.54514027]
 [ 5.66303253  2.94083999]
 [-0.63544568  3.56209868]
 [ 2.507542    9.13331259]
 [ 6.52046124  7.51951981]
 [ 4.11585764  3.24150618]
 [ 2.37346519  4.91463614]
 [ 8.73813113  7.97768912]
 [ 3.67828377  2.65075799]
 [ 6.14292306  8.99561043]
 [ 4.22240695  4.68333179]
 [ 1.42801693  3.77810175]
 [ 5.13627062  7.85058456]
 [ 5.52451457  5.98445687]
 [ 5.61811936  9.13487924]]
[0 0 1 0 1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1]

"""

X = np.transpose(X)
Y = Y.reshape((1, 1000))

print(X.shape)
print(Y.shape)
"""
(2, 1000)
(1, 1000)
"""

plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']))
plt.show()


model = Classification()
model.train(X=X, Y=Y, num_iter=300, learning_rate=0.2)

print(model.params)
""" {'W': array([[0.50008784, 0.51763796]]), 'b': array([[-4.7045604]])} """


plt.plot(model.cost)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iteration vs. Loss")
plt.show()


def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

    x_line = np.arange(np.min(X[0, :]), np.max(X[0, :]) * 1.1, 0.1)
    ax.plot(x_line, - W[0, 0] / W[0, 1] * x_line + -b[0, 0] / W[0, 1], color="black")
    plt.title("Classification (model3)")
    plt.plot()
    plt.show()


plot_decision_boundary(X, Y, model.params)



