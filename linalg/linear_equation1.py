import numpy as np
import matplotlib.pyplot as plt





# Representing and Solving System of Linear Equations using Matrices




# a simple system of equations (two equations) can be represented as a matrix:

"""
   -X1 + 3X2 = 7
   3X1 + 2X2 = 1
"""

mat1 = np.array(object=[[-1, 3], [3, 2]], dtype=np.dtype(float))  # this represents the coefficients

mat2 = np.array(object=[7, 1], dtype=np.dtype(float))  # this represents the answer of the equations

print(mat1)
print(mat2)

"""

[[-1.  3.]
 [ 3.  2.]]
 
[7. 1.]

"""





# check the dimension of mat1 & mat2

print(mat1.shape)
print(mat2.shape)

"""
(2, 2)
(2,)
"""

sol1 = np.linalg.solve(mat1, mat2)

print(sol1)

"""
[-1.  2.]

this means X1 = -1  & X2 = 2

"""






# calculate determinant of a matrix

det1 = np.linalg.det(mat1)

print(det1)

"""
-11.000000000000002
"""







# Solving System of Linear Equations using Elimination Method

"""
In the elimination method you either add or subtract the
equations of the linear system to get an equation with 
smaller number of variables. If needed, you can also 
multiply whole equation by non-zero number. 
"""

"""
Unify matrix 𝐴 and array 𝑏 into one matrix using np.hstack() function.
Note that the shape of the originally defined array 𝑏 was (2,), 
to stack it with the (2,2) matrix you need to use .reshape((2, 1)) function:
"""

mat3 = np.hstack((mat1, mat2.reshape((2, 1))))
print(mat3)

"""
[[-1.  3.  7.]
 [ 3.  2.  1.]]
"""




mat4 = mat3.copy()

mat4[1] = 3 * mat4[0] + mat4[1]

print(mat4)

mat4[1] = 1/11 * mat4[1]

print(mat4)

"""
[[-1.  3.  7.]
 [ 0.  1.  2.]]
 
so : 0X1 + X2 = 2 -> X2 = 2
so : -1X1 + (3 * 2) = 7 -> -X1 = 1 -> X1 = -1 
"""






# Graphical Representation of the Solution



def plot_lines(M):

    x_1 = np.linspace(-10, 10, 100)
    x_2_line_1 = (M[0, 2] - M[0, 0] * x_1) / M[0, 1]
    x_2_line_2 = (M[1, 2] - M[1, 0] * x_1) / M[1, 1]

    _, ax = plt.subplots(figsize=(10, 10))

    ax.plot(x_1, x_2_line_1, '-', linewidth=2, color='#0075ff',
            label=f'$x_2={-M[0, 0] / M[0, 1]:.2f}x_1 + {M[0, 2] / M[0, 1]:.2f}$')

    ax.plot(x_1, x_2_line_2, '-', linewidth=2, color='#ff7300',
            label=f'$x_2={-M[1, 0] / M[1, 1]:.2f}x_1 + {M[1, 2] / M[1, 1]:.2f}$')


    A = M[:, 0:-1]
    b = M[:, -1::].flatten()
    d = np.linalg.det(A)


    if d != 0:

        solution = np.linalg.solve(A, b)
        ax.plot(solution[0], solution[1], '-o', mfc='none',
                markersize=10, markeredgecolor='#ff0000', markeredgewidth=2)

        ax.text(solution[0] - 0.25, solution[1] + 0.75, f'$(${solution[0]:.0f}$,{solution[1]:.0f})$', fontsize=14)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))


    plt.xlabel('$x_1$', size=14)
    plt.ylabel('$x_2$', size=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()





plot_lines(mat3)







# System of Linear Equations with No Solutions

"""
   -X1 + 3X2 = 7
   3X1 - 9X2 = 1
"""

mat5 = np.array(object=[[-1, 3], [3, -9]], dtype=np.dtype(float))
mat6 = np.array(object=[7, 1], dtype=np.dtype(float))

det2 = np.linalg.det(mat5)

print(det2)
"""
0.0
"""

try:
    sol2 = np.linalg.solve(mat5, mat6)
    print(sol2)
except np.linalg.LinAlgError as err:
    print("Error: ", err)

"""
Error:  Singular matrix
"""



mat7 = np.hstack((mat5, mat6.reshape(2, 1)))
print(mat7)

"""
[[-1.  3.  7.]
 [ 3. -9.  1.]]
"""




mat8 = mat7.copy()

mat8[1] = 3 * mat8[0] + mat8[1]

print(mat8)

"""
[[-1.  3.  7.]
 [ 0.  0. 22.]]

in the second row: 0 != 22; so no solutions!

"""




plot_lines(mat7)






# System of Linear Equations with Infinite Number of Solutions


mat9 = np.array(object=[7, -21], dtype=np.dtype(float))

mat10 = np.hstack((mat5, mat9.reshape(2, 1)))

mat11 = mat10.copy()

mat11[1] = 3 * mat11[0] + mat11[1]

print(mat11)

"""
[[-1.  3.  7.]
 [ 0.  0.  0.]]

the entire second row is zero, so infinite solutions are possible!
"""

try:
    sol3 = np.linalg.solve(mat5, mat9)
    print(sol3)
except np.linalg.LinAlgError as err:
    print("Error: ", err)


"""
Error:  Singular matrix
"""





plot_lines(mat10)



