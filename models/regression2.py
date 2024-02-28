import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" A Linear Regression Model """
class Perceptron(object):
    def __init__(self, params=None, norm=True):

        if params:
            self.params = params
        else:
            self.params = {}

        self.norm = norm
        self.cost = []

    def  layer_size(self, x, y):  # Defining the Neural Network Structure

        in_layer = x.shape[0]
        out_layer = y.shape[0]

        return in_layer, out_layer

    def init_params(self, in_layer, out_layer, alpha, b):  # Initialize the Model's Parameters

        weight = np.random.randn(out_layer, in_layer) * alpha
        bias = np.zeros((out_layer, b))

        params = {"W": weight, "b": bias}

        return params

    def for_prop(self, x, w, b):  # Z = WX + b

        z = np.matmul(w, x) + b
        y_hat = z

        return y_hat

    def cost_func(self, y_hat, y):  # L(w, b) = sum(y_hat - y)² * 1/2m

        cost = np.sum(np.power(y_hat - y, 2)) / (2 * y_hat.shape[1])

        return cost


    def back_prop(self, y_hat, x, y):
        # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity.
        dz = y_hat - y
        dw = np.dot(dz, x.T) / x.shape[1]
        db = np.sum(dz, axis=1, keepdims=True) / x.shape[1]

        grads = {"dW": dw, "db": db}

        return grads

    def update(self, w, b, dw, db, l_rate):

        w = w - l_rate * dw
        b = b - l_rate * db

        return w, b

    def fit(self, x, y, alpha=1e-2, l_rate=1e-1, num_iter=100, b=1):

        in_layer, out_layer = self.layer_size(x=x, y=y)

        if len(self.params) == 0:

            self.params = self.init_params(in_layer=in_layer, out_layer=out_layer, alpha=alpha, b=b)

        if self.norm:
            x = (x - np.mean(x)) / np.std(x)
            y = (y - np.mean(y)) / np.std(y)

        for i in range(num_iter):

            y_hat = self.for_prop(x, self.params["W"], self.params["b"])
            grads = self.back_prop(y_hat=y_hat, x=x, y=y)
            cost = self.cost_func(y_hat=y_hat, y=y)
            w, b = self.update(w=self.params["W"], b=self.params["b"], dw=grads["dW"], db=grads["db"], l_rate=l_rate)

            self.cost.append(cost)
            self.params["W"] = w
            self.params["b"] = b

    def predict(self, x):
        if self.norm:
            x = (x - np.mean(x)) / np.std(x)

        predict = np.matmul(self.params["W"], x) + self.params["b"]

        return predict





path = "/home/sam/Documents/projects/machine_learning/data/tvmarketing.csv"

data = pd.read_csv(path)
print(data)
"""
     TV  Sales
0  230.1   22.1
1   44.5   10.4
2   17.2    9.3
3  151.5   18.5
4  180.8   12.9
"""

plt.scatter(data["TV"], data["Sales"])
plt.show()

X = data["TV"]
Y = data["Sales"]
print(X.shape)
print(Y.shape)
"""
(200,)
(200,)
"""

X = np.array(X).reshape((1, 200))
Y = np.array(Y).reshape((1, 200))
print(X.shape)
print(Y.shape)

X_train = X[:, :180]
Y_train = Y[:, :180]
X_test = X[:, 180:]
Y_test = Y[:, 180:]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
"""
(1, 180)
(1, 180)
(1, 20)
(1, 20)
"""

# train a model
model = Perceptron()
model.fit(x=X_train, y=Y_train, b=1, alpha=1e-2, l_rate=1e-2, num_iter=200)

print(model.params)
print(model.cost)
"""
{'W': array([[0.67446627]]), 'b': array([[5.56951661e-17]])}
[0.49467878866466564, 0.48876600652914687, ..., 0.20310639579302178, 0.2029959042756488]
"""


plt.plot(model.cost)
plt.show()


y_hat = model.predict(x=X_test)

print(y_hat)
print(Y_test)


# train a multiple linear model
path1 = "/home/sam/Documents/projects/machine_learning/data/house.csv"

data1 = pd.read_csv(path1)
print(data1.columns)
"""
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')
"""

X1 = data1[['GrLivArea', 'OverallQual']]
Y1 = data1['SalePrice']

print(X1.shape)
print(Y1.shape)
"""
(1460, 2)
(1460,)
"""

X1 = np.array(X1).T
Y1 = np.array(Y1).reshape((1, 1460))

print(X1.shape)
print(Y1.shape)
"""
(2, 1460)
(1, 1460)
"""

model1 = Perceptron()
model1.fit(x=X1, y=Y1, b=1, alpha=1e-2, l_rate=0.1, num_iter=100)

print(model1.params)
print(model1.cost)
"""
{'W': array([[1.05545105, 0.47139245]]), 'b': array([[-0.51256812]])}
"""

plt.plot(model1.cost)
plt.show()
