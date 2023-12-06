import numpy as np



# Representing and Solving a System of Linear Equations using Matrices


"""
    4X1 - 3X2 + X3  = -10
    2X1 + 2X2 + 3X3 = 0
    -X1 + 2X2 - 5X3 = 17
"""



mat1 = np.array(object=[[4, -3, 1], [2, 1, 3], [-1, 2, -5]], dtype=np.dtype(float))  # the left side of the equation (coefficients)

mat2 = np.array(object=[-10, 0, 17], dtype=np.dtype(float))  # the right side of the equation


print(mat1)
print(mat2)


"""

[[ 4. -3.  1.]
 [ 2.  1.  3.]
 [-1.  2. -5.]]
 
 
[-10.   0.  17.]

"""



print(np.shape(mat1))
print(np.shape(mat2))

"""
(3, 3)
(3,)
"""



sol1 = np.linalg.solve(mat1, mat2)

print(sol1)


"""
[ 1.  4. -2.]

so: X1 = 1
    X2 = 4
    X3 = -2
"""



det1 = np.linalg.det(mat1)

print(det1)



"""
determinant of the mat1 matrix: -60.000000000000036
 so the system of the equation is non-singular
"""






# Solving System of Linear Equations using Row Reduction



mat3 = np.hstack((mat1, mat2.reshape(3, 1)))

print(mat3)



"""

[[  4.  -3.   1. -10.]
 [  2.   1.   3.   0.]
 [ -1.   2.  -5.  17.]]
 
"""







# Multiply any row by a non-zero number


def row_multi(mat, row, num):

    mat_c = mat.copy()

    mat_c[row] = mat_c[row] * num

    return mat_c

print(mat3)



print(row_multi(mat=mat3, row=2, num=2))

"""

[[  4.  -3.   1. -10.]
 [  2.   1.   3.   0.]
 [ -1.   2.  -5.  17.]]
 
 
 
[[  4.  -3.   1. -10.]
 [  2.   1.   3.   0.]
 [ -2.   4. -10.  34.]]
 
"""






# Add two rows and exchange one of the original rows with the result of the addition
def row_addition(mat, row1, row2, num):

    mat_c = mat.copy()

    mat_c[row2] = num * mat_c[row1] + mat_c[row2]

    return mat_c




print(mat3)

print(row_addition(mat=mat3, row1=1, row2=2, num=0.5))



"""

[[  4.  -3.   1. -10.]
 [  2.   1.   3.   0.]
 [ -1.   2.  -5.  17.]]
 
 
 
[[  4.   -3.    1.  -10. ]
 [  2.    1.    3.    0. ]
 [  0.    2.5  -3.5  17. ]]
 
 
"""







# Swap rows
def row_swap(mat, row1, row2):

    mat_c = mat.copy()

    mat_c[[row1, row2]] = mat_c[[row2, row1]]

    return mat_c




print(mat3)

print(row_swap(mat3, row1=0, row2=2))



"""

[[  4.  -3.   1. -10.]
 [  2.   1.   3.   0.]
 [ -1.   2.  -5.  17.]]
 
 
 
[[ -1.   2.  -5.  17.]
 [  2.   1.   3.   0.]
 [  4.  -3.   1. -10.]]
 
"""




# Row Reduction and Solution of the Linear System

step1 = row_swap(mat=mat3, row1=0, row2=2)



print(step1)



"""
[[ -1.   2.  -5.  17.]
 [  2.   1.   3.   0.]
 [  4.  -3.   1. -10.]]
"""



step2 = row_addition(mat=step1, row1=0, row2=1, num=2)

print(step2)



"""
[[ -1.   2.  -5.  17.]
 [  0.   5.  -7.  34.]
 [  4.  -3.   1. -10.]]
"""




step3 = row_addition(mat=step2, row1=0, row2=2, num=4)

print(step3)



"""
[[ -1.   2.  -5.  17.]
 [  0.   5.  -7.  34.]
 [  0.   5. -19.  58.]]
"""




step4 = row_addition(mat=step3, row1=1, row2=2, num=-1)



print(step4)

"""
[[ -1.   2.  -5.  17.]
 [  0.   5.  -7.  34.]
 [  0.   0. -12.  24.]]
"""



step5 = row_multi(mat=step4, row=2, num=-1/12)

print(step5)



"""
[[-1.  2. -5. 17.]
 [ 0.  5. -7. 34.]
 [-0. -0.  1. -2.]]
"""




# so:

x_3 = -2

x_2 = (step5[1, 3] - step5[1, 2] * x_3) / step5[1, 1]

x_1 = (step5[0, 3] - step5[0, 2] * x_3 - step5[0, 1] * x_2) / step5[0, 0]

print(x_1, x_2, x_3)

"""
1.0  4.0  -2
"""





# System of Linear Equations with No Solutions

"""
    X1 + X2 + X3 = 2
    X2 - X3 = 1
    2X1 + X2 + 5X3 = 0
    
"""

mat4 = np.array(object=[[1, 1, 1], [0, 1, -3], [2, 1, 5]], dtype=np.dtype(float))

mat5 = np.array(object=[2, 1, 0], dtype=np.dtype(float))

det2 = np.linalg.det(mat4)




print(mat4)
print(mat5)
print(det2)

"""

[[ 1.  1.  1.]
 [ 0.  1. -3.]
 [ 2.  1.  5.]]
 
[2. 1. 0.]


0.0;  determinant is equal zero, so the system is singular.

"""



#sol2 = np.linalg.solve(mat4, mat5)

#print(sol2)

"""
raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
"""

mat6 = np.hstack((mat4, mat5.reshape(3, 1)))



step11 = row_addition(mat6, 0, 2, -2)
print(step11)

"""
[[ 1.  1.  1.  2.]
 [ 0.  1. -3.  1.]
 [ 0. -1.  3. -4.]]
"""



step12 = row_addition(step11, 1, 2, 1)
print(step12)

"""
[[ 1.  1.  1.  2.]
 [ 0.  1. -3.  1.]
 [ 0.  0.  0. -3.]]
"""





# System of Linear Equations with Infinite Number of Solutions


"""
   X1 + X2 + X3 = 2
   X2 - 3X3 = 1
   2X1 + X2 + 5X3 = 3
"""


mat7 = np.array([2, 1, 3])

mat8 = np.hstack((mat4, mat7.reshape((3, 1))))

print(mat8)


"""
[[ 1.  1.  1.  2.]
 [ 0.  1. -3.  1.]
 [ 2.  1.  5.  3.]]
"""




step21 = row_addition(mat8, 0, 2, -2)

print(step21)

"""
[[ 1.  1.  1.  2.]
 [ 0.  1. -3.  1.]
 [ 0. -1.  3. -1.]]
"""



step22 = row_addition(step21, 1, 2, 1)

print(step22)

"""
[[ 1.  1.  1.  2.]
 [ 0.  1. -3.  1.]
 [ 0.  0.  0.  0.]]
"""





