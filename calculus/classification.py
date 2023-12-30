import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs


# a single perceptron neural network model to solve a simple classification problem

np.random.seed(3)


fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.2
xline = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="b")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="b")

ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.set_title("Classification")
ax.plot(xline, xline + 0.5, color="black")
plt.show()


m = 100
X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))
print(X)
print(Y)
"""
[[0 0 1 1 0 0 0 1 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0 0 1 0 1 1 0
  1 0 0 1 1 0 0 1 0 1 0 1 1 1 1 0 1 0 0 1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 0 0
  1 1 1 0 1 1 1 1 1 1 0 1 0 0 1 1 0 1 0 0 0 0 0 1 1 0 1 1]
 [1 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 1 1 0
  0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 1 1 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1
  1 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 1 1]]
[[1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 1 0 0 0
  0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 1 1
  0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0]]
"""
print(X.shape)
print(Y.shape)
"""
(2, 100)
(1, 100)
"""


def sigmoid(Z):

  A = 1 / (1 + np.exp(-Z))

  return A


example = np.array([-4, -2, 0, 0.5, 8, 10])
print(sigmoid(example))
""" [0.01798621 0.11920292 0.5        0.62245933 0.99966465 0.9999546 ] """


def layer_size(X, Y):

  in_l, out_l = X.shape[0], Y.shape[0]

  return in_l, out_l


def init_params(in_l, out_l):

    W = np.random.randn(out_l, in_l) * 0.01
    b = np.zeros((out_l, 1))

    params = {"W": W, "b": b}

    return params


example2 = init_params(2, 1)
print(example2)
""" {'W': array([[0.01103922, 0.01881755]]), 'b': array([[0.]])} """


def for_prop(X, W, b):
    Z = np.matmul(W, X) + b

    return Z


def log_loss(A, Y):

  # loss = np.mean(-Y * np.log(A) - (1 - Y) * np.log(1 - A))

  log = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
  loss = np.mean(log)

  return loss

Z = for_prop(X, example2["W"], example2["b"])
A = sigmoid(Z)
loss = log_loss(A, Y)
print(A)
print(loss)

"""
[[0.49771979 0.49771979 0.5        0.49732965 0.49732965 0.49732965
  0.5        0.49771979 0.49771979 0.49771979 0.49732965 0.49771979
  0.5        0.5        0.49732965 0.49960985 0.49960985 0.49771979
  0.5        0.49771979 0.49771979 0.49771979 0.49732965 0.5
  0.49960985 0.49732965 0.5        0.5        0.49732965 0.49771979
  0.49960985 0.49771979 0.49732965 0.5        0.5        0.5
  0.49771979 0.5        0.5        0.49771979 0.5        0.49732965
  0.49771979 0.49732965 0.49960985 0.49960985 0.49771979 0.49771979
  0.49960985 0.5        0.49732965 0.5        0.49771979 0.49771979
  0.49732965 0.49960985 0.5        0.49771979 0.49732965 0.49732965
  0.5        0.49960985 0.49732965 0.5        0.5        0.49960985
  0.49960985 0.49960985 0.49771979 0.49732965 0.49960985 0.5
  0.5        0.49732965 0.5        0.5        0.49960985 0.49732965
  0.5        0.49771979 0.49732965 0.49960985 0.49960985 0.49771979
  0.49771979 0.49771979 0.49732965 0.49771979 0.49732965 0.49732965
  0.49960985 0.49771979 0.49732965 0.49732965 0.49960985 0.49732965
  0.49960985 0.49960985 0.5        0.5       ]]
0.6906898987378014

"""


def back_prop(A, Y, X):

  dZ = A - Y
  dW = 1/X.shape[1] * np.dot(dZ, X.T)
  db = 1/X.shape[1] * np.sum(dZ, axis=1, keepdims=True)

  grads = {"dW": dW, "db": db}

  return grads

grads = back_prop(A, Y, X)
print(grads["dW"])
print(grads["db"])
""" 
[[0.26768569 0.0328814 ]]
[[0.30362653]] 
"""


def update(W, b, dW, db, alpha):

  W -= dW * alpha
  b -= db * alpha

  params = {"W": W, "b": b}

  return params


def train(X, Y, num_iter=30, learning_rate=0.1):

  cost = []

  in_l, out_l = layer_size(X, Y)
  params = init_params(in_l, out_l)

  for i in range(num_iter):

    Z = for_prop(X, params["W"], params["b"])
    A = sigmoid(Z)
    grads = back_prop(A, Y, X)
    params = update(params["W"], params["b"], grads["dW"], grads["db"], learning_rate)
    cost.append(log_loss(A, Y))

  return params, cost

params, cost = train(X, Y, num_iter=50, learning_rate=1.2)
print(params)
print(cost)

"""
{'W': array([[-3.29428081,  2.8321211 ]]), 'b': array([[-1.79489309]])}
[0.691265943150345, 0.5391389493316354, ..., 0.13373039543575918, 0.1316933166028363]
"""


def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

    x_line = np.arange(np.min(X[0, :]), np.max(X[0, :]) * 1.1, 0.1)
    ax.plot(x_line, - W[0, 0] / W[0, 1] * x_line + -b[0, 0] / W[0, 1], color="black")
    plt.plot()
    plt.show()


plot_decision_boundary(X, Y, params)


def predict(X, params):

    Z = for_prop(X, params["W"], params["b"])
    A = sigmoid(Z)

    predictions = A > 0.5

    return predictions


X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, params)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")


X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, params)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")
"""
Coordinates (in the columns):
[[1 1 0 0]
 [0 1 0 1]]
Predictions:
[[False False False  True]]

"""


