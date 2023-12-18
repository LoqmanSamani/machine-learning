import numpy as np

mat1 = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])

print(mat1)

"""
[[4 9 9]
 [9 1 6]
 [9 2 3]]
"""

mat2 = np.array([[2, 2], [5, 7], [4, 4]])

print(mat2)

"""
[[2 2]
 [5 7]
 [4 4]]
"""

# matrix multiplication
mat3 = np.matmul(mat1, mat2)

print(mat3)

"""
[[ 89 107]
 [ 47  49]
 [ 40  44]]
"""

mat4 = mat1 @ mat2  # same operation as np.matmul(mat1, mat2)

print(mat4)

"""
[[ 89 107]
 [ 47  49]
 [ 40  44]]
"""

try:
    print(np.matmul(mat2, mat1))

except ValueError as err:
    print(err)

""" 
matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
"""

try:
    print(mat2 @ mat1)

except ValueError as err1:
    print(err1)

"""
matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)

"""


mat5 = np.array([1, -2, -5])

mat6 = np.array([4, 3, -1])

print(mat5.shape)  # (3,)
print(mat5.ndim)  # 1
print(mat5.reshape((3, 1)).shape)  # (3, 1)
print(mat5.reshape((3, 1)).ndim)  # 2

mat7 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3])

print(mat7)  # [1 2 3 4 5 6 7 8 9 1 2 3]

print(mat7.reshape((12, 1)))

"""
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]
 [1]
 [2]
 [3]]
"""

print(np.matmul(mat5, mat6))
""" 3 """

try:
    print(np.matmul(mat5.reshape((3, 1)), mat6.reshape(3, 1)))
except ValueError as err2:
    print(err2)
"""
matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 1)

"""


mat8 = np.dot(mat1, mat2)  # same operation as np.matmul(mat1, mat2)

print(mat8)

"""
[[ 89 107]
 [ 47  49]
 [ 40  44]]
"""
# What actually happens is what is called broadcasting in Python:
# NumPy broadcasts this dot product operation to all rows and all columns,
# you get the resultant product matrix. Broadcasting also works in other cases,
# for example:

print(mat1 - 2)

"""
[[ 2  7  7]
 [ 7 -1  4]
 [ 7  0  1]]
"""




