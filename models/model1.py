import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression




# a regression model with just one perceptron


class SingleRegressionPerceptron(object):

    def __init__(self, learning_rate=1e-2, max_iter=50):

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.params = {}
        self.cost = []

    def layer_size(self, X, Y):

        input_layer = X.shape[0]
        output_layer = Y.shape[0]

        return input_layer, output_layer

    def param_init(self, input_layer, output_layer):

        params = {}
        W = np.random.randn(output_layer, input_layer)
        b = np.zeros((output_layer, 1))

        params["W"] = W * 0.01
        params["b"] = b

        return params

    def forward_propagation(self, X, params):

        W = params["W"]
        b = params["b"]

        y_hat = np.dot(W, X) + b

        return y_hat

    def compute_cost(self, Y, y_hat):

        m = Y.shape[1]

        cost = np.sum(np.power(y_hat - Y, 2)) / (2 * m)

        return cost

    def backward_propagation(self, y_hat, X, Y):

        grads = {}
        m = X.shape[1]

        dZ = y_hat - Y
        dW = 1 / m * np.matmul(dZ, X.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        grads["dW"] = dW
        grads["db"] = db

        return grads

    def param_update(self, grads):

        params = {}

        W = self.params["W"]
        b = self.params["b"]

        dW = grads["dW"]
        db = grads["db"]

        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db

        params["W"] = W
        params["b"] = b

        return params



    def train(self, X, Y):

        input_layer, output_layer = self.layer_size(X=X, Y=Y)
        self.params = self.param_init(
            input_layer=input_layer,
            output_layer=output_layer
        )

        for i in range(self.max_iter):

            y_hat = self.forward_propagation(
                X=X,
                params=self.params
            )

            cost = self.compute_cost(
                Y=Y,
                y_hat=y_hat
            )

            grads = self.backward_propagation(
                y_hat=y_hat,
                X=X,
                Y=Y
            )

            params = self.param_update(
                grads=grads
            )

            self.params = params
            self.cost.append(cost)







np.random.seed(42)



# generate random data (1000 points) for a simple linear regression model
X, Y = make_regression(
    n_samples=1000,
    n_features=1,
    bias=0.75,
    noise=20,
    random_state=1
)

print(X.shape)
""" (1000, 1) """
print(Y.shape)
""" (1000,) """

X = X.reshape((1, 1000))
Y = Y.reshape((1, 1000))

print(X.shape)
""" (1, 1000) """
print(Y.shape)
""" (1, 1000) """


# visualize the generated data

plt.scatter(X, Y, c="black")
plt.ylabel("Y s")
plt.xlabel("X s")
plt.title("Regression points (model 1)")
plt.show()



# split data into train and test sets
X_train = X[:, :900]
Y_train = Y[:, :900]
X_test = X[:, 900:]
Y_test = Y[:, 900:]

print(X_test)
print(Y_test)
model = SingleRegressionPerceptron(learning_rate=1e-2, max_iter=400)
model.train(X_train, Y_train)

print(model.params)
""" 
{'W': array([[37.41944349]]), 'b': array([[1.20030062]])}
"""
print(model.cost)
"""
[897.7408781861071, 884.5363043281623, ..., 201.86483043566363, 201.85774622328242, 201.8507936279864]
"""


iterations = range(1, len(model.cost) + 1)

plt.plot(iterations, model.cost, label="Training Cost")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title("Change of Training Cost Over Iterations (model 1)")
plt.legend()
plt.show()

# optimal W and b
weight = 37.41944349
bias = 1.20030062

# prediction

y_hat = (weight * X_test) + bias


plt.scatter(X_test, Y_test, c="blue", label="Actual Data")
plt.scatter(X_test, y_hat, c="green", label="Predicted Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Values (model 1)")

plt.legend()
plt.show()




# generate random data (1000 points) for a multiple linear regression model
X1, Y1 = make_regression(
    n_samples=1000,
    n_features=2,
    bias=0.5,
    noise=50,
    random_state=1
)



print(X1.shape)
""" (1000, 2) """
print(Y1.shape)
""" (1000,) """




X1 = X1.reshape((2, 1000))
Y1 = Y1.reshape((1, 1000))


print(X1.shape)
""" (2, 1000) """
print(Y1.shape)
""" (1, 1000) """




# visualize the generated data

plt.scatter(X1[0], Y1, c="red", label="Feature 1")
plt.scatter(X1[1], Y1, c="green", label="Feature 2")
plt.ylabel("Y s")
plt.xlabel("X s")
plt.title("Regression points (model 2)")
plt.show()





# split data into train and test sets
X_train1 = X1[:, :900]
Y_train1 = Y1[:, :900]
X_test1 = X1[:, 900:]
Y_test1 = Y1[:, 900:]


print(X_test1)
print(Y_test1)

model = SingleRegressionPerceptron(learning_rate=1e-2, max_iter=400)
model.train(X_train1, Y_train1)


print(model.params)
""" 
{'W': array([[1.75200639, 2.11981615]]), 'b': array([[4.18819714]])}
"""
print(model.cost)
"""
[5189.745576679497, 5189.463123689479, ..., 5176.203807372199, 5176.203740995953, 5176.203675999639]
"""



iterations = range(1, len(model.cost) + 1)

plt.plot(iterations, model.cost, label="Training Cost")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title("Change of Training Cost Over Iterations1 (model 1)")
plt.legend()
plt.show()




# optimal W and b
weights = np.array([1.75200639, 2.11981615])
weights = weights.reshape((1, 2))
bias1 = 4.18819714




# prediction

y_hat1 = np.matmul(weights, X_test1) + bias1


plt.scatter(X_test1[0, :], Y_test1, c="blue", label="Actual Data (Feature 1)")
plt.scatter(X_test1[0, :], y_hat1, c="red", label="Predicted Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Values 1 (model 1)")

plt.legend()
plt.show()


plt.scatter(X_test1[1, :], Y_test1, c="black", label="Actual Data (Feature 1)")
plt.scatter(X_test1[1, :], y_hat1, c="red", label="Predicted Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Values 1 (model 1)")

plt.legend()
plt.show()

