import numpy as np



mat1 = np.array(object=
                [[2, -1, 1, 1],
                 [1, 2, -1, -1],
                 [-1, 2, 2, 2],
                 [1, -1, 2, 1]],
                dtype=np.dtype(float))


mat2 = np.array(object=
                [6, 3, 14, 8],
                dtype=np.dtype(float))



print(mat1.shape)
print(mat2.shape)




det1 = np.linalg.det(mat1)

print(det1)


sol1 = np.linalg.solve(mat1, mat2)

print(sol1)




# Elementary Operations and Row Reduction

# row multiplication
def row_mul(mat, row, num):

    new_m = mat.copy()

    new_m[row] = new_m[row] * num

    return new_m



# row addition
def row_add(mat, row1, row2, num):

    new_m = mat.copy()

    new_m[row2] = (new_m[row1] * num) + new_m[row2]

    return new_m




# swap rows

def row_swap(mat, row1, row2):

    new_mat = mat.copy()

    new_mat[[row1, row2]] = new_mat[[row2, row1]]

    return new_mat




mat3 = np.array(object=[
        [1, -2, 3, -4],
        [-5, 6, -7, 8],
        [-4, 3, -2, 1],
        [8, -7, 6, -5]
    ], dtype=np.dtype(float))





print(row_mul(mat3, 2, -2))

print(row_add(mat3, 0, 2, 4))

print(row_swap(mat3, 0, 2))



print(np.array([1,2,3] + np.array([3,5,6])))







def row_reduction(mat1, mat2):


    new_mat = np.hstack((mat1, mat2.reshape(4, 1)))

    new_mat = row_swap(new_mat, 0, 1)

    new_mat = row_add(new_mat, 0, 1, -2)

    new_mat = row_add(new_mat, 0, 2, 1)

    new_mat = row_add(new_mat, 0, 3, -1)

    new_mat = row_add(new_mat, 2, 3, 1)

    new_mat = row_swap(new_mat, 1, 3)

    new_mat = row_add(new_mat, 2, 3, 1)

    new_mat = row_add(new_mat, 1, 2, -4)

    new_mat = row_add(new_mat, 1, 3, 1)

    new_mat = row_add(new_mat, 3, 2, 2)

    new_mat = row_add(new_mat, 2, 3, -8)

    new_mat = row_mul(new_mat, 3, -1 / 17)


    return new_mat




mat4 = row_reduction(mat1, mat2)

print(mat4)






# find the value of x_4
x_4 = mat4[3, 4]

# find the value of x_3
x_3 = mat4[2, 4] - (3 * mat4[3, 4])

# find the value of x_2
x_2 = mat4[1, 4] - (4 * x_3) - (3 * x_4)

# find the value of x_1
x_1 = mat4[0, 4] - (2 * x_2) + x_3 + x_4


print(x_1, x_2, x_3, x_4)






def row_reduction1(mat):


    new_mat = row_add(mat, 3,2, -3)

    new_mat = row_add(new_mat, 3, 1, -3)

    new_mat = row_add(new_mat, 3, 0, 1)

    new_mat = row_add(new_mat, 2, 1, -4)

    new_mat = row_add(new_mat, 2, 0, 1)

    new_mat = row_add(new_mat, 1, 0, -2)


    return new_mat


mat5 = row_reduction1(mat4)

print(mat5)

