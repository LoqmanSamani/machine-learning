```python
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
import numpy as np
import pandas as pd
import sympy
import math
%matplotlib inline
```

### functions in python
### f(x) = x²


```python
def power_f(x):
    result = x**2

    return result


print(power_f(3))
```

    9



```python
def der_f(x):
    return 2*x
```

### Derivative of exponent terms


```python
def derivative(base, power):
    return f"{power} {base}^{power - 1}"

derivative(base="X", power=2)
```




    '2 X^1'




```python
derivative(base="Y", power=3)

```




    '3 Y^2'




```python
""" 3 Y^2 """

mat1 = np.array([2, 3, 5, 6])

power_f(mat1)
```




    array([ 4,  9, 25, 36])




```python
derivative(base=mat1, power=2)
```




    '2 [2 3 5 6]^1'



### plot function and its derivative 


```python
def plot_funcs(power_f, der_f=None, x_min=-5, x_max=5, label1="f(x)", label2="f'(x)"):

    x = np.linspace(x_min, x_max, 100)

    # Setting the axes at the centre.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, power_f(x), 'r', label=label1)
    if not der_f is None:
        # If f2 is an array, it is passed as it is to be plotted as unlinked points.
        # If f2 is a function, f2(x) needs to be passed to plot it.
        if isinstance(der_f, np.ndarray):
            plt.plot(x, der_f, 'bo', markersize=3, label=label2, )
        else:
            plt.plot(x, der_f(x), 'b', label=label2)
    plt.legend()

    plt.show()


plot_funcs(power_f, der_f)
```


    
![png](output_10_0.png)
    


### Symbolic Computation with `sympy`



```python
approx = math.sqrt(18)
approx
```




    4.242640687119285




```python
exact = sympy.sqrt(18)
exact
```




$\displaystyle 3 \sqrt{2}$




```python
approx1 = sympy.N(sympy.sqrt(18), n=8)
approx1
```




$\displaystyle 4.2426407$




```python
x, y = sympy.symbols("x y")

sympy.expr = 2 * x ** 2 - x * y
```


```python
x
```




$\displaystyle x$




```python
y
```




$\displaystyle y$




```python
sympy.expr
```




$\displaystyle 2 x^{2} - x y$




```python
sympy.expr_manip = x * (sympy.expr + x * y + x**3)
sympy.expr_manip
```




$\displaystyle x \left(x^{3} + 2 x^{2}\right)$




```python
sympy.expand(sympy.expr_manip)
```




$\displaystyle x^{4} + 2 x^{3}$




```python
sympy.factor(sympy.expr_manip)
```




$\displaystyle x^{3} \left(x + 2\right)$




```python
sympy.expr.evalf(subs={x:-1, y:2})
```




$\displaystyle 4.0$



### evaluate a function 𝑓(𝑥)=𝑥2


```python
sympy.f_symb = x ** 2
sympy.f_symb.evalf(subs={x:3})
```




$\displaystyle 9.0$




```python
print(mat1)
```

    [2 3 5 6]



```python
try:
    sympy.f_symb(mat1)
except TypeError as err:
    print(err)
```

    'Pow' object is not callable



```python
f_symb_numpy = lambdify(x, sympy.f_symb, 'numpy')
```


```python
print(mat1)
print(f_symb_numpy(mat1))
```

    [2 3 5 6]
    [ 4  9 25 36]


### Symbolic Differentiation with `SymPy`


```python
sympy.diff(x**3,x)
```




$\displaystyle 3 x^{2}$




```python
dfdx_composed = sympy.diff(sympy.exp(-2*x) + 3*sympy.sin(3*x), x)
dfdx_composed
```




$\displaystyle 9 \cos{\left(3 x \right)} - 2 e^{- 2 x}$




```python
dfdx_symb = sympy.diff(sympy.f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')
```


```python
print(mat1)
print(dfdx_symb_numpy(mat1))
```

    [2 3 5 6]
    [ 4  6 10 12]



```python
plot_funcs(f_symb_numpy, dfdx_symb_numpy)
```


    
![png](output_34_0.png)
    


### Limitations of Symbolic Differentiation


```python
dfdx_abs = sympy.diff(abs(x),x)
dfdx_abs
```




$\displaystyle \frac{\left(\operatorname{re}{\left(x\right)} \frac{d}{d x} \operatorname{re}{\left(x\right)} + \operatorname{im}{\left(x\right)} \frac{d}{d x} \operatorname{im}{\left(x\right)}\right) \operatorname{sign}{\left(x \right)}}{x}$




```python
dfdx_abs.evalf(subs={x:-2})
```




$\displaystyle - \left. \frac{d}{d x} \operatorname{re}{\left(x\right)} \right|_{\substack{ x=-2 }}$




```python
dfdx_abs_numpy = lambdify(x, dfdx_abs,'numpy')

try:
    dfdx_abs_numpy(np.array([1, -2, 0]))
except NameError as err:
    print(err)
```

    name 'Derivative' is not defined


### Numerical Differentiation with NumPy

#### find the derivative of function 𝑓(𝑥)=𝑥2 defined above. The first argument is an array of function values, the second defines the spacing Δ𝑥 for the evaluation.


```python
mat2 = np.linspace(-5, 5, 100)
plt.plot(mat2)
plt.show()
```


    
![png](output_41_0.png)
    



```python
mat2_ = power_f(mat2)
plt.plot(mat2_)
plt.show()
```


    
![png](output_42_0.png)
    



```python
grads = np.gradient(mat2_, mat2)
plt.plot(grads)
plt.show()
```


    
![png](output_43_0.png)
    



```python
plot_funcs(dfdx_symb_numpy, grads, label1="f'(x) exact", label2="f'(x) approximate")
```


    
![png](output_44_0.png)
    


#### neumerical differentiation  of more complicated function


```python
def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

mat2__ = f_composed(mat2)
plt.plot(mat2__)
plt.show()

#plot_f1_and_f2(lambdify(x, dfdx_composed, 'numpy'), np.gradient(f_composed(x_array_2), x_array_2),
             # label1="f'(x) exact", label2="f'(x) approximate")
```


    
![png](output_46_0.png)
    



```python
grads_ = np.gradient(mat2__, mat2)
plt.plot(grads_)
plt.show()
```


    
![png](output_47_0.png)
    



```python
plot_funcs(lambdify(x, dfdx_composed, 'numpy'), grads_, label1="f'(x) exact", label2="f'(x) approximate")
```


    
![png](output_48_0.png)
    


#### Limitations of Numerical Differentiation


```python
def dfdx_abs(x):
    if x > 0:
        return 1
    else:
        if x < 0:
            return -1
        else:
            return None

plot_funcs(np.vectorize(dfdx_abs), np.gradient(abs(mat2), mat2))
```


    
![png](output_50_0.png)
    


####  the results near the "jump" are 0.5 and −0.5, while they should be 1 and −1. These cases can give significant errors in the computations

#### Automatic Differentiation with `jax`

Automatic differentiation (autodiff) method breaks down the function into common functions (𝑠𝑖𝑛, 𝑐𝑜𝑠, 𝑙𝑜𝑔, power functions, etc.), and constructs the computational graph consisting of the basic functions. Then the chain rule is used to compute the derivative at any node of the graph. It is the most commonly used approach in machine learning applications and neural networks, as the computational graph for the function and its derivatives can be built during the construction of the neural network, saving in future computations.

The main disadvantage of it is implementational difficulty. However, nowadays there are libraries that are convenient to use, such as MyGrad, Autograd and JAX. Autograd and JAX are the most commonly used in the frameworks to build neural networks. JAX brings together Autograd functionality for optimization problems, and XLA (Accelerated Linear Algebra) compiler for parallel computing.


```python
from jax import grad, vmap
from jax import numpy as jnp 
```


```python
mat3 = jnp.array([2, 4, 5, 6, 9], dtype=np.dtype(float))
print(type(mat3))
print(type(mat2))
```

    <class 'jaxlib.xla_extension.ArrayImpl'>
    <class 'numpy.ndarray'>


    /tmp/ipykernel_13711/1854932869.py:1: UserWarning: Explicitly requested dtype float64 requested in array is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more.
      mat3 = jnp.array([2, 4, 5, 6, 9], dtype=np.dtype(float))



```python
print(mat3.dtype)
```

    float32



```python
mat3 = mat3.at[3].set(1999.0)
print(mat3)
```

    [   2.    4.    5. 1999.    9.]


The following code will calculate the derivative of the previously defined function 𝑓(𝑥)=𝑥2 at the point 𝑥=3:


```python
print(power_f(3))
print(grad(power_f)(3.0))
```

    9
    6.0



```python
try:
    grad(power_f)(3)
except TypeError as err:
    print(err)
```

    grad requires real- or complex-valued inputs (input dtype that is a sub-dtype of np.inexact), but got int32. If you want to use Boolean- or integer-valued inputs, use vjp or set allow_int to True.



```python
try:
    grad(power_f)(mat3)
except TypeError as err:
    print(err)
```

    Gradient only defined for scalar-output functions. Output had shape: (5,).



```python
print(vmap(power_f)(mat3))
```

    [4.000000e+00 1.600000e+01 2.500000e+01 3.996001e+06 8.100000e+01]



```python
plot_funcs(power_f, vmap(grad(power_f)))
```


    
![png](output_63_0.png)
    


#### Computational Efficiency of Symbolic, Numerical and Automatic Differentiation


```python
import timeit, time

x_array_large = np.linspace(-5, 5, 1000000)

tic_symb = time.time()
res_symb = lambdify(x, sympy.diff(power_f(x),x),'numpy')(x_array_large)
toc_symb = time.time()
time_symb = 1000 * (toc_symb - tic_symb)  # Time in ms.

tic_numerical = time.time()
res_numerical = np.gradient(power_f(x_array_large),x_array_large)
toc_numerical = time.time()
time_numerical = 1000 * (toc_numerical - tic_numerical)

tic_jax = time.time()
res_jax = vmap(grad(power_f))(jnp.array(x_array_large.astype('float32')))
toc_jax = time.time()
time_jax = 1000 * (toc_jax - tic_jax)

print(f"Results\nSymbolic Differentiation:\n{res_symb}\n" + 
      f"Numerical Differentiation:\n{res_numerical}\n" + 
      f"Automatic Differentiation:\n{res_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_symb} ms\n" + 
      f"Numerical Differentiation:\n{time_numerical} ms\n" + 
      f"Automatic Differentiation:\n{time_jax} ms")
```

    Results
    Symbolic Differentiation:
    [-10.       -9.99998  -9.99996 ...   9.99996   9.99998  10.     ]
    Numerical Differentiation:
    [-9.99999 -9.99998 -9.99996 ...  9.99996  9.99998  9.99999]
    Automatic Differentiation:
    [-10.       -9.99998  -9.99996 ...   9.99996   9.99998  10.     ]
    
    
    Time
    Symbolic Differentiation:
    3.747701644897461 ms
    Numerical Differentiation:
    36.76867485046387 ms
    Automatic Differentiation:
    4.716634750366211 ms



```python
def f_polynomial_simple(x):
    return 2*x**3 - 3*x**2 + 5

def f_polynomial(x):
    for i in range(3):
        x = f_polynomial_simple(x)
    return x

tic_polynomial_symb = time.time()
res_polynomial_symb = lambdify(x, sympy.diff(f_polynomial(x),x),'numpy')(x_array_large)
toc_polynomial_symb = time.time()
time_polynomial_symb = 1000 * (toc_polynomial_symb - tic_polynomial_symb)

tic_polynomial_jax = time.time()
res_polynomial_jax = vmap(grad(f_polynomial))(jnp.array(x_array_large.astype('float32')))
toc_polynomial_jax = time.time()
time_polynomial_jax = 1000 * (toc_polynomial_jax - tic_polynomial_jax)

print(f"Results\nSymbolic Differentiation:\n{res_polynomial_symb}\n" + 
      f"Automatic Differentiation:\n{res_polynomial_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_polynomial_symb} ms\n" +  
      f"Automatic Differentiation:\n{time_polynomial_jax} ms")
```

    Results
    Symbolic Differentiation:
    [2.88570423e+24 2.88556400e+24 2.88542377e+24 ... 1.86202587e+22
     1.86213384e+22 1.86224181e+22]
    Automatic Differentiation:
    [2.8857043e+24 2.8855642e+24 2.8854241e+24 ... 1.8620253e+22 1.8621349e+22
     1.8622416e+22]
    
    
    Time
    Symbolic Differentiation:
    285.7627868652344 ms
    Automatic Differentiation:
    25.6955623626709 ms

