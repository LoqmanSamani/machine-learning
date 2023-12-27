import numpy as np
import matplotlib.pyplot as plt


# gradient descent method for two variables

# function with one global minimum
# f(x, y) = 85 - 1/90 x²(x-6) y²(y-6)
def func1(x, y):

    func = 85 - ((1 / 90) * np.power(x, 2) * (x - 6) * np.power(y, 2) * (y - 6))

    return func


# derivative of the function with respect to both x and y
def der_func1x(x, y):

    der_fx = (-1/90) * x * (3 * x - 12) * np.power(y, 2) * (y - 6)

    return der_fx


def der_func1y(x, y):

    der_fy = (-1 / 90) * np.power(x, 2) * (x - 6) * y * (3 * y - 12)

    return der_fy


x2 = np.linspace(0, 5, 100)
y1 = np.linspace(0, 5, 100)
x_1, y_1 = np.meshgrid(x2, y1)  # used to create a 2D grid of coordinates from the 1D arrays x1 and y1
z_1 = func1(x_1, y_1)
print(z_1)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_1, y_1, z_1, cmap='coolwarm')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# implement gradient descent algorithm
def grad_descent(x, y, der_x, der_y, alpha=1e-2, iters=100):

    xs = []
    ys = []

    for i in range(iters):

        x, y = x - (alpha * der_x(x, y)), y - (alpha * der_y(x, y))
        xs.append(x)
        ys.append(y)

    return x, y, xs, ys


x, y, xs, ys = grad_descent(x=0.5, y=0.6, der_x=der_func1x, der_y=der_func1y, alpha=0.1, iters=50)

print(x)
""" 3.9996026579863093 """

print(y)
""" 3.9996064969422704  """

plt.plot([i for i in range(50)], xs, c="red", label="Xs")
plt.plot([i for i in range(50)], ys, c="blue", label="Ys")
plt.xlabel("Iteration")
plt.ylabel("Converging")
plt.title("Convergence Through Iterations")
plt.legend()
plt.show()


# function with multiple minima

def func2(x,y):

    func = -(10/(3+3*(x-.5)**2+3*(y-.5)**2) + \
            2/(1+2*((x-3)**2)+2*(y-1.5)**2) + \
            3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2))+10

    return func


def der_func2x(x,y):

    func = -(-2*3*(x-0.5)*10/(3+3*(x-0.5)**2+3*(y-0.5)**2)**2 + \
            -2*2*(x-3)*2/(1+2*((x-3)**2)+2*(y-1.5)**2)**2 +\
            -2*0.5*(x-3.5)*3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2)**2)

    return func


def der_func2y(x,y):

    func = -(-2*3*(y-0.5)*10/(3+3*(x-0.5)**2+3*(y-0.5)**2)**2 + \
            -2*2*(y-1.5)*2/(1+2*((x-3)**2)+2*(y-1.5)**2)**2 +\
            -0.5*2*(y-4)*3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2)**2)

    return func



x2 = np.linspace(0, 5, 100)
y2 = np.linspace(0, 5, 100)
x_2, y_2 = np.meshgrid(x2, y2)  # used to create a 2D grid of coordinates from the 1D arrays x1 and y1
z_2 = func2(x_2, y_2)
print(z_2)

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_2, y_2, z_2, cmap='coolwarm')

# Set the view from above
# ax.view_init(elev=90, azim=0)

ax.set_xlabel('X2')
ax.set_ylabel('Y2')
ax.set_zlabel('Z2')
ax.set_title('3D Plot')
plt.show()




# Local minimum

x3, y3, x3s, y3s = grad_descent(x=0.5, y=4, der_x=der_func2x, der_y=der_func2y, alpha=0.1, iters=70)

print(x3)
print(y3)
"""
3.4788712434347753
3.947504287482117
"""


plt.plot([i for i in range(70)], x3s, c="red", label="X3s")
plt.plot([i for i in range(70)], y3s, c="blue", label="Y3s")
plt.xlabel("Iteration")
plt.ylabel("Converging")
plt.title("Convergence Through Iterations")
plt.legend()
plt.show()



# Global minimum

x3, y3, x3s, y3s = grad_descent(x=0.5, y=3, der_x=der_func2x, der_y=der_func2y, alpha=0.1, iters=70)

print(x3)
print(y3)
"""
0.523032257935883
0.5169891562802666
"""


plt.plot([i for i in range(70)], x3s, c="red", label="X3s")
plt.plot([i for i in range(70)], y3s, c="blue", label="Y3s")
plt.xlabel("Iteration")
plt.ylabel("Converging")
plt.title("Convergence Through Iterations")
plt.legend()
plt.show()



# Local minimum

x3, y3, x3s, y3s = grad_descent(x=2, y=5, der_x=der_func2x, der_y=der_func2y, alpha=0.1, iters=70)

print(x3)
print(y3)
"""
3.478880403618373
3.947515173677233
"""

plt.plot([i for i in range(70)], x3s, c="red", label="X3s")
plt.plot([i for i in range(70)], y3s, c="blue", label="Y3s")
plt.xlabel("Iteration")
plt.ylabel("Converging")
plt.title("Convergence Through Iterations")
plt.legend()
plt.show()





