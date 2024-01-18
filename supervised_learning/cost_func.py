import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    y_hat = w * x + b
    cost = np.mean(np.sum((y_hat - y)**2)) / 2
    return cost


np.random.seed(43)
w = 200
b = 100
x = np.linspace(0, 10, 500)
y = w * x + b
y1 = np.array([np.random.randn()*400 + y[i] for i in range(len(x))])

ws = np.array(range(175, 226))
bs = np.array(range(75, 126))


j_wb = np.zeros((len(ws), len(bs)))
for i, w in enumerate(ws):
    for j, b in enumerate(bs):
        cost = compute_cost(x, y1, w, b)
        j_wb[i, j] = cost


W, B = np.meshgrid(ws, bs)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, j_wb, cmap='viridis')

ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('Cost')

plt.show()


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])


ws1 = np.array([i for i in range(-100, 500)])
bs1 = np.array([i for i in range(-300, 300)])
print(len(ws1))
print(len(bs1))
"""
600
600
"""

y_hat = np.zeros((len(ws1), len(bs1)))

for i, w in enumerate(ws):
    for j, b in enumerate(bs):
        cost1 = compute_cost(x_train, y_train, w, b)
        y_hat[i, j] = cost1


w1, b1 = np.meshgrid(ws1, bs1)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, b1, y_hat , cmap='coolwarm')
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('Cost')
ax.set_label("Cost Function")

plt.show()

