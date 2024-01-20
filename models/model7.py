# A Regression Model with batch gradient descent algorithm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Regression:

    def __init__(self, alpha=1e-2, norm_x=None, norm_y=None, max_iter=400, params=None, print_cost=100, sigma=1e-2, threshold=1e-8, threshold1=1e-12, epsilon=1e-8):

        self.alpha = alpha
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.max_iter = max_iter
        self.params = params
        self.print_cost = print_cost
        self.sigma = sigma
        self.threshold = threshold
        self.threshold1 = threshold1
        self.epsilon = epsilon

        self.params_ = []
        self.cost = []
        self.params = {}

    def layer_size(self, X, Y):

        in_layer = X.shape[0]
        out_layer = Y.shape[0]

        return in_layer, out_layer

    def param_init(self, in_layer, out_layer, sigma):

        W = np.random.randn(out_layer, in_layer) * sigma
        b = np.zeros((out_layer, 1))
        params = {"W": W, "b": b}

        return params

    def compute_cost(self, Y, Z):

        m = Y.shape[1]
        cost = np.sum(np.power(Y - Z, 2)) / (2 * m)
        return cost

    def forward_propagation(self, X, W, b):

        Z = np.matmul(W, X) + b
        return Z

    def backward_propagation(self, X, Y, Z):

        m = X.shape[1]
        dZ = Z - Y

        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        grads = {"dW": dW, "db": db}

        return grads

    def param_update(self, W, b, dW, db, alpha):

        W -= dW * alpha
        b -= db * alpha
        params = {"W": W, "b": b}

        return params

    def train(self, X, Y):

        if self.norm_x:
            mean_x = np.mean(X, axis=1, keepdims=True)
            std_x = np.std(X, axis=1, keepdims=True)
            X = (X - mean_x) / (std_x + self.epsilon)
        if self.norm_y:
            mean_y = np.mean(Y, axis=1, keepdims=True)
            std_y = np.std(Y, axis=1, keepdims=True)
            Y = (Y - mean_y) / (std_y + self.epsilon)

        X = np.transpose(X)
        Y = np.transpose(Y)

        in_layer, out_layer = self.layer_size(X, Y)
        self.params = self.param_init(in_layer, out_layer, self.sigma)

        for i in range(self.max_iter):
            Z = self.forward_propagation(X, self.params["W"], self.params["b"])
            cost = self.compute_cost(Y, Z)
            self.cost.append(cost)
            grads = self.backward_propagation(X, Y, Z)
            params = self.param_update(self.params["W"], self.params["b"], grads["dW"], grads["db"], self.alpha)

            if (params["W"] < self.threshold).all() and (params["b"] < self.threshold).all():
                self.params_.append(params)
                self.params = params
                if i % self.print_cost == 0:
                    print(f"Iteration {i}: {cost}")
                    print(f"after {i} iteration model is converged!")
                break

            elif cost < self.threshold1:
                if i % self.print_cost == 0:
                    print(f"Iteration {i}: {cost}")
                    print(f"after {i} iteration model is converged!")
                break

            else:
                self.params_.append(params)
                self.params = params
                if i % self.print_cost == 0:
                    print(f"Iteration {i}: {cost}")


    def predict(self, X):

        X = np.transpose(X)
        y_hat = np.matmul(self.params["W"], X) + self.params["b"]

        return y_hat



# Train the model
path = "/home/sam/Documents/projects/machine_learning/data/house.csv"

with open(path) as file:
    data = pd.read_csv(file)

print(data.columns)

X = np.array(data[['GrLivArea', 'OverallQual', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']])
Y = np.array(data['SalePrice'])

print(X.shape)
print(Y.shape)
"""
(1460, 5)
(1460,)
"""
Y = Y.reshape((1460, 1))


plt.figure(figsize=(8, 4))
plt.scatter(data['GrLivArea'], Y, color="red", label="X1")
plt.scatter(data['OverallQual'], Y, color="green", label="X2")
plt.scatter(data['BedroomAbvGr'], Y, color="blue", label="X3")
plt.scatter(data['KitchenAbvGr'], Y, color="black", label="X4")
plt.scatter(data['YrSold'], Y, color="brown", label="X5")
plt.xlabel("Feature")
plt.ylabel("House Price")
plt.title("Features vs. House Price (no normalization)")
plt.legend()
plt.show()


mean_x = np.mean(X, axis=1, keepdims=True)
std_x = np.std(X, axis=1, keepdims=True)
X = (X - mean_x) / (std_x + 1e-8)

mean_y = np.mean(Y, axis=1, keepdims=True)
std_y = np.std(Y, axis=1, keepdims=True)
Y = (Y - mean_y) / (std_y + 1e-8)

plt.figure(figsize=(8, 4))
plt.scatter(X[:, 0], Y, color="red", label="X1")
plt.scatter(X[:, 1], Y, color="green", label="X2")
plt.scatter(X[:, 2], Y, color="blue", label="X3")
plt.scatter(X[:, 3], Y, color="black", label="X4")
plt.scatter(X[:, 4], Y, color="brown", label="X5")
plt.xlabel("Feature")
plt.ylabel("House Price")
plt.title("Features vs. House Price (no normalization)")
plt.legend()
plt.show()


model = Regression(
    alpha=1e-3,
    norm_x=True,
    norm_y=True,
    max_iter=10000,
    params=None,
    print_cost=100,
    sigma=1e-2,
    threshold=1e-8,
    threshold1=1e-14,
    epsilon=1e-8
)

model.train(X, Y)
"""
Iteration 0: 3.667438585895999e-05
Iteration 100: 2.49370693360803e-05
Iteration 200: 2.0820964918787813e-05
Iteration 300: 1.9090554091441358e-05
Iteration 400: 1.811744649319557e-05
Iteration 500: 1.7394804643408276e-05
Iteration 600: 1.6764524458844477e-05
Iteration 700: 1.6176948414242265e-05
Iteration 800: 1.5616154579559403e-05
Iteration 900: 1.5076726780378898e-05
Iteration 1000: 1.4556531367457924e-05
Iteration 1100: 1.4054470805174909e-05
Iteration 1200: 1.3569784679132385e-05
Iteration 1300: 1.3101831848103353e-05
Iteration 1400: 1.2650022211908197e-05
Iteration 1500: 1.2213795015293017e-05
Iteration 1600: 1.1792611637448837e-05
Iteration 1700: 1.138595290145327e-05
Iteration 1800: 1.0993317804525056e-05
Iteration 1900: 1.0614222704978913e-05
Iteration 2000: 1.0248200665601444e-05
Iteration 2100: 9.894800859588083e-06
Iteration 2200: 9.553588009377082e-06
Iteration 2300: 9.224141848694368e-06
Iteration 2400: 8.906056604349496e-06
Iteration 2500: 8.598940496274772e-06
Iteration 2600: 8.302415254921509e-06
Iteration 2700: 8.016115655334913e-06
Iteration 2800: 7.739689067307548e-06
Iteration 2900: 7.472795021049215e-06
Iteration 3000: 7.215104787835647e-06
Iteration 3100: 6.966300975118989e-06
Iteration 3200: 6.726077135600797e-06
Iteration 3300: 6.494137389786306e-06
Iteration 3400: 6.270196061554748e-06
Iteration 3500: 6.053977326297016e-06
Iteration 3600: 5.845214871187302e-06
Iteration 3700: 5.643651567170344e-06
Iteration 3800: 5.449039152259889e-06
Iteration 3900: 5.261137925758831e-06
Iteration 4000: 5.079716453023941e-06
Iteration 4100: 4.90455128041192e-06
Iteration 4200: 4.7354266600554435e-06
Iteration 4300: 4.572134284130246e-06
Iteration 4400: 4.414473028285946e-06
Iteration 4500: 4.262248703924775e-06
Iteration 4600: 4.115273819022805e-06
Iteration 4700: 3.973367347199301e-06
Iteration 4800: 3.8363545047497075e-06
Iteration 4900: 3.704066535367547e-06
Iteration 5000: 3.5763405022902563e-06
Iteration 5100: 3.453019087612801e-06
Iteration 5200: 3.3339503985219515e-06
Iteration 5300: 3.2189877802125482e-06
Iteration 5400: 3.1079896352553215e-06
Iteration 5500: 3.000819249193824e-06
Iteration 5600: 2.8973446221555983e-06
Iteration 5700: 2.797438306270243e-06
Iteration 5800: 2.7009772486940644e-06
Iteration 5900: 2.6078426400480665e-06
Iteration 6000: 2.5179197680824813e-06
Iteration 6100: 2.4310978763877164e-06
Iteration 6200: 2.347270027977569e-06
Iteration 6300: 2.2663329735768276e-06
Iteration 6400: 2.188187024450964e-06
Iteration 6500: 2.112735929621188e-06
Iteration 6600: 2.0398867573138476e-06
Iteration 6700: 1.9695497804979687e-06
Iteration 6800: 1.9016383663700202e-06
Iteration 6900: 1.8360688696498148e-06
Iteration 7000: 1.7727605295560558e-06
Iteration 7100: 1.711635370334699e-06
Iteration 7200: 1.652618105217579e-06
Iteration 7300: 1.5956360436930032e-06
Iteration 7400: 1.5406190019741039e-06
Iteration 7500: 1.4874992165547e-06
Iteration 7600: 1.4362112607461637e-06
Iteration 7700: 1.386691964092509e-06
Iteration 7800: 1.3388803345644346e-06
Iteration 7900: 1.2927174834365114e-06
Iteration 8000: 1.2481465527549665e-06
Iteration 8100: 1.205112645306713e-06
Iteration 8200: 1.1635627570034293e-06
Iteration 8300: 1.1234457115973353e-06
Iteration 8400: 1.084712097648305e-06
Iteration 8500: 1.0473142076646423e-06
Iteration 8600: 1.0112059793426137e-06
Iteration 8700: 9.763429388323028e-07
Iteration 8800: 9.426821459599309e-07
Iteration 8900: 9.10182141339211e-07
Iteration 9000: 8.788028953065382e-07
Iteration 9100: 8.485057586171374e-07
Iteration 9200: 8.192534148414746e-07
Iteration 9300: 7.910098344032646e-07
Iteration 9400: 7.637402302024557e-07
Iteration 9500: 7.374110147685648e-07
Iteration 9600: 7.119897588915951e-07
Iteration 9700: 6.874451516795149e-07
Iteration 9800: 6.637469619931874e-07
Iteration 9900: 6.408660012112149e-07

"""

plt.plot(model.cost)
plt.show()

del model


