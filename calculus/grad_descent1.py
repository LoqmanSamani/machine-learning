import numpy as np
import matplotlib.pyplot as plt



# Optimization Using Gradient Descent in One Variable



# f(x) = e^x - log(x)
def function1(x):
    func1 = np.exp(x) - np.log(x)
    return func1



# f'(x) = e^x - 1/x
def der_function1(x):
    der_func1 = np.exp(x) - (1/x)
    return der_func1



# plot the function1
x_vals = np.linspace(start=0, stop=2.5, num=1000)
plt.figure(figsize=(8, 4))
plt.plot(x_vals, function1(x_vals))
plt.xlabel("X")
plt.ylabel("f(X)")
plt.title("f(X) = e^x - log(x)")
plt.xlim(0, 2.5)
plt.ylim(0, 12)
plt.show()





# Function with Multiple Minima (f(x) = 2*sin(x) - x²/10 + 1)
def function2(x):
    func2 = (2 * np.sin(x)) - (np.power(x, 2) / 10) + 1
    return func2


def der_function2(x):
    der_func2 = (2 * np.cos(x)) - (x/5)
    return der_func2





x2 = np.linspace(-8, 8, 200)


x3 = function2(x2)
print(x3[55])
print(x3[110])
print(x3[150])
print(x3[185])




plt.plot(x3)
plt.xlabel("X")
plt.ylabel("f(X)")
plt.title("f(x) = 2 * sin(x) - x²/10 + 1")
plt.show()




# gradient descent function
def gradient_descent(x, der_fx, learning_rate=1e-2, num_iter=100):

    xs = []
    for i in range(num_iter):
        x = x - learning_rate * der_fx(x)
        xs.append(x)
    return x, xs




x, xs = gradient_descent(x=1.6, der_fx=der_function1, num_iter=200)


print(x)
""" 0.5671434156768685 """

print(xs)
""" [1.5567196757560489, 1.5157110764355792, 1.4767820672520853, 0.567227356089708, 0.5672232605855196] """

plt.plot(xs)
plt.xlabel("Iteration")
plt.ylabel("X")
plt.title("Gradient Descent")
plt.show()






x11, xs1 = gradient_descent(x=-3.5, der_fx=der_function2, learning_rate=0.01, num_iter=300)
x22, xs2 = gradient_descent(x=0.7, der_fx=der_function2, learning_rate=0.01, num_iter=300)
x33, xs3 = gradient_descent(x=5, der_fx=der_function2, learning_rate=0.01, num_iter=300)
x44, xs4 = gradient_descent(x=7, der_fx=der_function2, learning_rate=0.01, num_iter=300)

print(x11)
print(x22)
print(x33)
print(x44)

plt.plot(xs1, label="X = -3.5")
plt.plot(xs2, label="X = 0.7")
plt.plot(xs3, label="X = 5")
plt.plot(xs4, label="X = 7")
plt.xlabel("Iteration")
plt.ylabel("X")
plt.title("Gradient Descent")
plt.legend()
plt.show()


