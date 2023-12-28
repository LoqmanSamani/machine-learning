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




