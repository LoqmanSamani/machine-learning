import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Optimization Using Gradient Descent: Linear Regression

path = "/home/sam/Documents/projects/machine_learning/data/tvmarketing.csv"

data = pd.read_csv(path)
print(data)
"""
       TV  Sales
0    230.1   22.1
1     44.5   10.4
2     17.2    9.3
3    151.5   18.5
4    180.8   12.9
..     ...    ...
195   38.2    7.6
196   94.2    9.7
197  177.0   12.8
198  283.6   25.5
199  232.1   13.4

"""

print(len(data))
""" 200 """


# Visualize the data
plt.scatter(data.TV, data.Sales)
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("TV vs. Sale")
plt.legend()
plt.show()


# using pandas to visualize the data
data.plot(x='TV', y='Sales', kind='scatter', c='black')


# Linear Regression with NumPy

X = data.TV
Y = data.Sales

W1, b = np.polyfit(x=X, y=Y, deg=1)

print(W1)
""" 0.04753664043301975 """
print(b)
""" 7.0325935491276965 """


# Visualize the linear regression model
def lr_plot(x, y, x_label, y_label, title, w, b, color1="black", color2="red", X_pred=np.array([]), Y_pred=np.array([]), color3="blue"):


    w1 = np.array([w]).reshape((1, np.array([w]).shape[0]))
    x1 = np.array(x).reshape((w1.shape[1], len(x)))
    y_hat = np.matmul(w1, x1) + b

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, y, 'o', color=color1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot(x, y_hat.reshape((len(x), 1)), color=color2)
    ax.plot(X_pred, Y_pred, 'o', color=color3, markersize=8)
    plt.show()

lr_plot(X, Y, "TV", "Sale", "LR using numpy",W1, b)


x_p = np.array([40, 140, 250])
y_p = W1 * x_p + b

lr_plot(X, Y, "TV", "Sale", "LR using numpy",W1, b, X_pred=x_p, Y_pred=y_p)



# Linear Regression with Scikit-Learn

model = LinearRegression()

print(X.shape)
print(Y.shape)
"""
(200,)
(200,)
"""
x3 = np.array(X).reshape((len(X), 1))
y3 = np.array(Y).reshape((len(Y), 1))

print(x3.shape)
print(y3.shape)
"""
(200, 1)
(200, 1)
"""


model.fit(x3, y3)

print(model.coef_)
print(model.intercept_)

"""
[[0.04753664]]
[7.03259355]
"""


# Linear regression using gradient descent


def cost_func(x, y, m, b):

    loss = 0.5 * np.mean(np.power(m * x + b - y, 2))

    return loss


def der_func_dm(x, y, m, b):

    grad = np.dot(m * x + b - y, x) / len(x)

    return grad


def der_func_db(x, y, m, b):

    grad = np.mean(m * x + b - y)

    return grad


def update(x, y, m, b, num_iter=100, alpha=1e-2, norm=True):

    cost = []

    # Z-Score Normalization: x - mean / std
    if norm:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    for i in range(num_iter):
        loss = cost_func(x=x, y=y, m=m, b=b)
        m_grad = der_func_dm(x=x, y=y, m=m, b=b)
        b_grad = der_func_db(x=x, y=y, m=m, b=b)

        m -= alpha * m_grad
        b -= alpha * b_grad
        cost.append(loss)

    return m, b, cost


slope, intercept, cost = update(x=X, y=Y, m=0, b=0, alpha=1.2, num_iter=30, norm=True)

print(slope)
print(intercept)
print(cost)
"""
0.7822244248616065
-3.19744231092045e-16
[0.5, 0.2062999755919659, 0.19455197461564455, ..., 0.19406247457496448, 0.19406247457496448]
"""

plt.plot(cost)
plt.show()

slope1, intercept1, cost1 = update(x=X, y=Y, m=0, b=0, num_iter=200)

print(slope1)
print(intercept1)
print(cost1)
"""
0.677422250752692
-2.0099477637813844e-16
[0.5, 0.4939118432440418, 0.48794484080752715,..., 0.19977949587404065, 0.19966572715018904]
"""

plt.plot(cost1)
plt.show()

slope2, intercept2, cost2 = update(x=X, y=Y, m=2, b=2, alpha=0.1, num_iter=30)

print(slope2)
print(intercept2)
print(cost2)
"""
0.8338473420109904
0.08478231655043211
[2.9355511502767877, 2.414668301893441,..., 0.20157121316258064, 0.20014455283093355] 
"""

plt.plot(cost2)
plt.show()






