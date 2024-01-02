import numpy as np
import matplotlib.pyplot as plt

def function(X, Y):

    fXY = (- 2 * np.power(X, 2)) + (- 3 * np.power(Y, 2)) - (X * Y) + 15

    return fXY

X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)

X1, Y1 = np.meshgrid(X, Y)
Z1 = function(X1, Y1)



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, Y1, Z1, cmap='coolwarm')

# Set the view from above
# ax.view_init(elev=90, azim=0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('-2X² + -3Y² - XY + 15')
plt.show()



