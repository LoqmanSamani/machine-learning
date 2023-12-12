import numpy as np
import cv2


# a simple vector transformation function
def transform1(vec):

    new_vec = np.zeros((3, 1))

    new_vec[0, 0] = 3 * vec[0, 0]
    new_vec[2, 0] = -2 * vec[1, 0]

    return new_vec

mat1 = np.array([[3], [5]])


print(mat1)

"""
[[3]
 [5]]
"""

print(transform1(mat1))

"""
[[  9.]
 [  0.]
 [-10.]]
"""


# Transformations Defined as a Matrix Multiplication

def transform2(vec1):

    mat = np.array([[3, 0], [0, 0], [0, -2]])

    transformed = mat @ vec1

    return transformed

mat2 = np.array([[3], [5]])

mat3 = transform2(mat2)
print(mat3)

"""
[[  9]
 [  0]
 [-10]]
"""


"""
Every linear transformation can be carried out by matrix multiplication. 
And vice versa, carrying out matrix multiplication, it is natural to 
consider the linear transformation that it represents.
"""


# Standard Transformations in a Plane









