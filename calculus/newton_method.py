import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# function with one variable

# f(x) = e^x - log(x)
def func(X):

    f_X = np.exp(X) - np.log(X)

    return f_X

# f'(x) = e^x - 1/x
def der_func(X):

    der_fX = np.exp(X) - (1/ X)

    return der_fX

# f"(x) = e^x + 1/X²
def der2_func(X):

    der2_fX = np.exp(X) + (1/np.power(X, 2))

    return der2_fX


X2 = 1.6
print(func(X2))
print(der_func(X2))
print(der2_func(X2))
"""
4.483028795149379
4.328032424395115
5.343657424395115
"""


def plot_f(x_range, y_range, f, ox_position):
    x = np.linspace(*x_range, 100)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)
    ax.set_ylabel('$f\,(x)$')
    ax.set_xlabel('$x$')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position(('data', ox_position))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.autoscale(enable=False)

    pf = ax.plot(x, f(x), 'k')
    plt.show()
    return fig, ax


# plot_f([0.001, 2.5], [-0.3, 13], func, 0.0)


# implement newton's method
def newton_method(X, der_func, der2_func, num_iter=20):

    Xs = [X]
    for i in range(num_iter):

        X = X - der_func(X) / der2_func(X)
        Xs.append(X)

    return X, Xs

x, xs = newton_method(X=1.6, der_func=der_func, der2_func=der2_func, num_iter=20)

print(x)
print(xs)
"""
0.5671432904097838
[1.6, 0.7900617721793732, 0.5436324685389214, 0.5665913613835818, 0.567143002403454, ..., 0.5671432904097838] 
"""


def gradient_descent(X1, der_func, num_iter=20, alpha=1e-2):

    Xs1 = [X1]
    for i in range(num_iter):
        dx1 = der_func(X2)
        X1 -= dx1 * alpha
        Xs1.append(X1)

    return X1, Xs1


x1, xs1 = gradient_descent(X1=1.6, der_func=der_func, num_iter=20, alpha=0.01)

print(x1)
print(xs1)


plt.plot(xs, label="Newton's Method", color="red")
plt.plot(xs1, label="Gradient Descent Method", color="green")
plt.xlabel("Iteration")
plt.ylabel("X")
plt.title("Newton's Method vs. Gradient Descent")
plt.legend()
plt.show()

"""
Those are disadvantages of gradient descent method in comparison with Newton's method: 
  - there is an extra parameter to control (learning rate)
  - it converges slower
However it has an advantage:
  - in each step you do not need to calculate second derivative, 
    which in more complicated cases is quite computationally expensive to find. 
    So, one step of gradient descent method is easier to make than one step of Newton's method.
"""


# Function in Two Variables

# f(x, y) = x⁴ + 0.8Y⁴ + 4X² + 2Y² − XY − 0.2X²Y
def func1(X, Y):

    result = np.power(X, 4) + (0.8 * np.power(Y, 4)) + (4 * np.power(X, 2)) + (2 * np.power(Y, 2)) - (X * Y) - (0.2 * np.power(X, 2) * Y)

    return result

# calculate gradient-vector
def der_func1(X, Y):

    df_dx = (4 * np.power(X, 3)) + (8 * X) - Y + (0.4 * X * Y)
    df_dy = (3.2 * np.power(Y, 3)) + (4 * Y) - X - (0.2 * np.power(X, 2))

    grads = np.array([df_dx, df_dy])

    return grads


# calculate Hessian matrix
def hessian(X, Y):

    HM = np.zeros((2, 2))

    HM[0, 0] = (12 * np.power(X, 2)) + 8 + (0.4 * Y)
    HM[0, 1] = -1 - (0.4 * X)
    HM[1, 0] = -1 - (0.4 * X)
    HM[1, 1] = (9.6 * np.power(Y, 2)) + 4

    return HM

def newton_method1(X, Y, der_func, der2_func, num_iter=20):

    XYs = np.zeros((2, num_iter))
    XYs[0, 0] = X
    XYs[1, 0] = Y

    for i in range(1, num_iter):

        grads = der_func(X, Y)
        HM = der2_func(X, Y)
        new_xy = XYs[:, i-1] - np.linalg.inv(HM) @ grads
        XYs[0, i] = new_xy[0]
        XYs[1, i] = new_xy[1]
        X = XYs[0, i]
        Y = XYs[1, i]

    return XYs, X, Y


print(func1(4, 4))
print(der_func1(4, 4))
print(hessian(4, 4))
"""
528.0

[290.4 213.6]

[[201.6  -2.6]
 [ -2.6 157.6]]
"""

X2 = np.linspace(-4, 4, 100)
Y2 = np.linspace(-4, 4, 100)

X3, Y3 = np.meshgrid(X2, Y2)
Z3 = func1(X3, Y3)



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X3, Y3, Z3, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title("f(x, y) = x⁴ + 0.8Y⁴ + 4X² + 2Y² − XY − 0.2X²Y")
plt.show()


xys2, x2, y2 = newton_method1(4, 4, der_func1, hessian)

print(xys2)
print(x2)
print(y2)



def gradient_descent2(X, Y, der_func, num_iter=20, alpha=1e-2):

    XYs = np.zeros((2, num_iter))
    XYs[0, 0] = X
    XYs[1, 0] = Y

    for i in range(1, num_iter):
        grads = der_func(X, Y)
        X = X - alpha * grads[0]
        Y = Y - alpha * grads[1]
        XYs[0, i] = X
        XYs[1, i] = Y

    return XYs, X, Y


xys4, x4, y4 = gradient_descent2(4, 4, der_func1, num_iter=20, alpha=0.01)


plt.plot(xys2[0, :], label="NM X")
plt.plot(xys2[1, :], label="NM Y")
plt.plot(xys4[0, :], label="GD X")
plt.plot(xys4[1, :], label="GD Y")
plt.xlabel("Iteration")
plt.ylabel("X & Y")
plt.title("NM vs. GD with two Variables")
plt.legend()
plt.show()






