import numpy as np
import matplotlib.pyplot as plt
import time





# visualization of a vector
def plot_vectors(list_v, list_label, list_color):


    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.axis([-10, 10, -10, 10])
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0] - 0.2 + sgn[0], v[1] - 0.2 + sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()




v = np.array([[1], [3]])
# Arguments: list of vectors as NumPy arrays, labels, colors.
# plot_vectors([v], [f"$v$"], ["black"])

# plot_vectors([v, 2*v, -2*v], [f"$v$", f"$2v$", f"$-2v$"], ["black", "green", "blue"])


vec1 = np.array([[1], [3]])
vec2 = np.array([[4], [-1]])

# plot_vectors([vec1, vec2, vec1 + vec2], [f"$vec1$", f"$vec2$", f"$vec1 + vec2$"], ["black", "black", "red"])


vec3 = np.array([[2], [3]])


# calculate the norm (magnitude or length) of vectors
print(np.linalg.norm(vec1))
print(np.linalg.norm(vec2))
print(np.linalg.norm(vec3))

"""
3.1622776601683795
4.123105625617661
3.605551275463989
"""



# dot product of two vectors

"""
   vec1 * vec2 = sum(vec1 * vec2) = vec1(1) * vec2(1) + vec1(2) * vec2(2) + ... + vec1(n) * vec2(n)
"""




def dot_product(vector1, vector2):

    result = 0
    if len(vector1) == len(vector2):

        for i in range(len(vector1)):
            mul = vector1[i] * vector2[i]
            result += mul

    else:
        print("the input vectors must be the sae length!!!")

    return result




lst1 = [1, -2, -5]
lst2 = [4, 3, -1]


print(dot_product(vector1=lst1, vector2=lst2))

""" 3 """



print(np.dot(lst1, lst2))  # dot function works with list as input also

""" 3 """



print(np.array(lst1) @ np.array(lst2))  # @ function works only with array inputs

""" 3 """



# speed of Calculations in Vectorized Form




vec4 = np.random.rand(1000000)
vec5 = np.random.rand(1000000)




t1 = time.time()
dot1 = dot_product(vector1=vec4, vector2=vec5)
t2 = time.time()

print(t2 - t1)

""" 0.22990918159484863 """  # the slowest method

t3 = time.time()
dot2 = np.dot(vec4, vec5)
t4 = time.time()

print(t4 - t3)

""" 0.0010228157043457031 """




t5 = time.time()
dot3 = vec4 @ vec5
t6 = time.time()

print(t6 - t5)

""" 0.0009975433349609375 """  # the fastest method






vec6 = np.array([1, 0, 0])
vec7 = np.array([0, 1, 0])

print(dot_product(vec6, vec7))

""" 
0  so it means the angle between vec6 and vec7 are 90 degree (vec6 and vec7 are orthogonal)

because  dot product: vec1 * vec2 = |vec1| * |vec2| * cos(theta)

"""
