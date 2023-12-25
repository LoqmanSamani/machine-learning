from jax import grad  # used to perform automatic differentiation
import jax.numpy as jnp  # integrated numpy in jax library
import matplotlib.pyplot as plt
import pandas as pd





path = "/home/sam/Documents/projects/machine_learning/data/supplier_price.csv"

data = pd.read_csv(path)

print(data)
"""
      date     price_supplier_a_dollars_per_item    price_supplier_b_dollars_per_item 
0   1/02/2016                                104                                   76  
1   1/03/2016                                108                                   76
2   1/04/2016                                101                                   84
3   1/05/2016                                104                                   79
4   1/06/2016                                102                                   81                       
"""

print(len(data))
"""  50 """

print(data.columns)
"""
Index(['date', 'price_supplier_a_dollars_per_item',
       'price_supplier_b_dollars_per_item'],
       dtype='object')
"""

print(data.date)
"""
0     1/02/2016
1     1/03/2016
...
48    1/02/2020
49    1/03/2020
"""

price_a = jnp.array(object=data["price_supplier_a_dollars_per_item"], dtype="float32")
print(price_a)
"""
[104. 108. 101. 104. 102. 105. 114. 102. 105. 101. 109. 103.  93.  98.
  92.  97.  96.  94.  97.  93.  99.  93.  98.  94.  93.  92.  96.  98.
  98.  93.  97. 102. 103. 100. 100. 104. 100. 103. 104. 101. 102. 100.
 102. 108. 107. 107. 103. 109. 108. 108.]
"""
price_b = jnp.array(object=data["price_supplier_b_dollars_per_item"], dtype="float32")
print(price_b)
"""
[ 76.  76.  84.  79.  81.  84.  90.  93.  93.  99.  98.  96.  94. 104.
 101. 102. 104. 106. 105. 103. 106. 104. 113. 115. 114. 124. 119. 115.
 112. 111. 106. 107. 108. 108. 102. 104. 101. 101. 100. 103. 106. 100.
  97.  98.  90.  92.  92.  99.  94.  91.]
"""

"""
prices_A = None(None).astype('None')
prices_B = None(None).astype('None')
"""

plt.plot(price_a, label="Prices A")
plt.plot(price_b, label="Prices B")
plt.xlabel("Time (month)")
plt.ylabel("Price")
plt.title("Price Through Month")
plt.legend()
plt.show()




# Calculate f_of_omega, corresponding to the 𝑓𝑖(𝜔)=𝑝𝑖𝐴𝜔+𝑝𝑖𝐵(1−𝜔).
def f_of_omega(omega, pA, pB):

    f = (pA * omega) + (pB * (1 - omega))
    return f




# Calculate calculate L_of_omega L(𝜔) = 1/k * sum((𝑓𝑖(𝜔) - mean(𝑓𝑖(𝜔)))**2)
def l_of_omega(omega, pA, pB):

    f = f_of_omega(omega=omega, pA=pA, pB=pB)
    l = jnp.mean(jnp.power((f - jnp.mean(f)), 2))

    return l





print("L(omega = 0) =", l_of_omega(0, price_a, price_b))
print("L(omega = 0.2) =", l_of_omega(0.2, price_a, price_b))
print("L(omega = 0.8) =", l_of_omega(0.8, price_a, price_b))
print("L(omega = 1) =", l_of_omega(1, price_a, price_b))
"""
L(omega = 0) = 110.72
L(omega = 0.2) = 61.1568
L(omega = 0.8) = 11.212797
L(omega = 1) = 27.48
"""




def opt_omega(pA, pB, num_iter):

    cost = jnp.inf
    omega = 0
    learning_rate = 1 / num_iter

    for i in range(num_iter):
        l = l_of_omega(i * learning_rate, pA, pB)
        if l < cost:
            cost = l
            omega = i * learning_rate

    return cost, omega



cost, omega = opt_omega(pA=price_a, pB=price_b, num_iter=1000)



print(cost)
""" 9.24972 """
print(omega)
""" 0.7020000000000001 """





# Another way to do the same

N = 1001
omega_array = jnp.linspace(0, 1, N, endpoint=True)

def L_of_omega_array(omega_array, pA, pB):
    N = len(omega_array)
    L_array = jnp.zeros(N)

    for i in range(N):

        L = l_of_omega(omega_array[i], pA, pB)
        L_array = L_array.at[i].set(L)

    return L_array




L_array = L_of_omega_array(omega_array, price_a, price_b)

i_opt = L_array.argmin()
omega_opt = omega_array[i_opt]
L_opt = L_array[i_opt]

print(f'omega_min = {omega_opt:.3f}\nL_of_omega_min = {L_opt:.7f}')
"""
omega_min = 0.702
L_of_omega_min = 9.2497196
"""




# Calculate dL/d𝜔 for each 𝜔 in the omega_array

def der_l_array(N, pA, pB):

    derivatives = jnp.zeros(N)
    omega_array = jnp.linspace(0, 1, N, endpoint=True)

    for i in range(N):
        dLdOmega = grad(l_of_omega)(omega_array[i], pA, pB)
        derivatives = derivatives.at[i].set(dLdOmega)

    return derivatives


array = der_l_array(1001, price_a, price_b)
print(array)
"""
[-288.96     -288.54858  -288.13712  ...  121.65716   122.068596 122.47998 ]
"""

print(array[690: 710])

"""
[-5.066391   -4.6549377  -4.243469   -3.8320923  -3.4205627  -3.0091705
 -2.5977478  -2.1862946  -1.7748718  -1.3634186  -0.9519806  -0.5405426
 -0.12905884  0.2823639   0.6937561   1.1052551   1.5166626   1.9280701
  2.3395386   2.7510376 ]
"""



# or

i_opt_2 = jnp.abs(array).argmin()
omega_opt_2 = omega_array[i_opt_2]
dLdOmega_opt_2 = array[i_opt_2]
print(f'omega_min = {omega_opt_2:.3f}\ndLdOmega_min = {dLdOmega_opt_2:.7f}')

"""
omega_min = 0.702
dLdOmega_min = -0.1290588
"""


# Visualization of the L(𝜔) and L'(𝜔) functions
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Setting the axes at the origin.
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(omega_array,  L_array, "black", label = "$\mathcal{L}\\left(\omega\\right)$")
plt.plot(omega_array,  array, "orange", label = "$\mathcal{L}\'\\left(\omega\\right)$")
plt.plot([omega_opt, omega_opt_2], [L_opt,dLdOmega_opt_2], 'ro', markersize=3)

plt.legend()

plt.show()



