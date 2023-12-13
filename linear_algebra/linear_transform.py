import numpy as np
import matplotlib.pyplot as plt
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

mat4 = np.array([[2], [-3], [9]])

mat5 = np.array([[2], [5], [1]])

num = 6


result1, result2 = transform1(vec=num * mat5), num * transform1(vec=mat5)
result3, result4 = transform1(vec=mat4 + mat5), transform1(vec=mat4) + transform1(vec=mat5)



print(f"T(num * mat5): {result1}")
print(f"T(mat5) * num: {result2}")
print(f"T(mat4 * mat5): {result3}")
print(f"T(mat4) + T(mat5: {result4}")

"""

T(num * mat5): 
[[ 36.]
 [  0.]
 [-60.]]

T(mat5) * num: 
[[ 36.]
 [  0.]
 [-60.]]

T(mat4 * mat5): 
[[12.]
 [ 0.]
 [-4.]]

T(mat4) + T(mat5: 
[[12.]
 [ 0.]
 [-4.]]

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

mat6 = np.array([[1], [0]])

mat7 = np.array([[0], [1]])

print(transform2(vec1=mat6))

"""
[[3]
 [0]
 [0]]
"""

print(transform2(vec1=mat7))

"""
[[ 0]
 [ 0]
 [-2]]
"""

mat8 = np.hstack((mat6, mat7))


print(mat8)
"""
[[1 0]
 [0 1]]
"""

print(mat8.shape)

""" (2, 2) """

mat9 = transform2(vec1=mat8)

print(mat9)

"""
[[ 3  0]
 [ 0  0]
 [ 0 -2]]

"""


# Horizontal Scaling (Dilation)




def h_scal_transform(vec2):

    mat = np.array([[2, 0], [0, 1]])
    result = mat @ vec2

    return result





def transform_vector(transform, vec3, vec4):

    result5 = np.hstack((vec3, vec4))
    result6 = transform(result5)

    return result6





mat10 = np.array([[1], [0]])

mat11 = np.array([[0], [1]])




result7 = transform_vector(transform=h_scal_transform, vec3=mat10, vec4=mat11)



print(f"Original Vector 1 : {mat10}")

print(f"Original Vector 2 : {mat11}")

print(f"Transformed Matrix: {result7}")


"""

Original Vector 1 : 
[[1]
 [0]]
 
Original Vector 2 : 
[[0]
 [1]]
 
Transformed Matrix: 
[[2 0]
 [0 1]]
 
"""




def transform_plot(transform, vec1, vec2):

    color1 = "#129cab"
    color2 = "#cc8933"

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))

    plt.axis([-5, 5, -5, 5])
    plt.quiver([0, 0], [0, 0], [vec1[0], vec2[0]], [vec1[1], vec2[1]], color=color1, angles='xy', scale_units='xy',
               scale=1)
    plt.plot([0, vec2[0], vec1[0], vec1[0]],
             [0, vec2[1], vec2[1], vec1[1]],
             color=color1)
    e1_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(vec1)])
    ax.text(vec1[0] - 0.2 + e1_sgn[0], vec1[1] - 0.2 + e1_sgn[1], f'$e_1$', fontsize=14, color=color1)
    e2_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(vec2)])
    ax.text(vec2[0] - 0.2 + e2_sgn[0], vec2[1] - 0.2 + e2_sgn[1], f'$e_2$', fontsize=14, color=color1)

    vec1_t = transform(vec1)
    vec2_t = transform(vec2)

    plt.quiver([0, 0], [0, 0], [vec1_t[0], vec2_t[0]], [vec1_t[1], vec2_t[1]],
               color=color2, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, vec2_t[0], vec1_t[0] + vec2_t[0], vec1_t[0]],
             [0, vec2_t[1], vec1_t[1] + vec2_t[1], vec1_t[1]],
             color=color2)
    e1_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(vec1_t)])
    ax.text(vec1_t[0] - 0.2 + e1_transformed_sgn[0], vec1_t[1] - e1_transformed_sgn[1],
            f'$T(e_1)$', fontsize=14, color=color2)
    e2_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(vec2_t)])
    ax.text(vec2_t[0] - 0.2 + e2_transformed_sgn[0], vec2_t[1] - e2_transformed_sgn[1],
            f'$T(e_2)$', fontsize=14, color=color2)

    plt.gca().set_aspect("equal")
    plt.show()




transform_plot(transform=h_scal_transform, vec1=mat10, vec2=mat11)





# Reflection about y-axis (the vertical axis)

def reflection_yaxis(vec):

    mat = np.array([[-1, 0], [0, 1]])
    transformed = mat @ vec

    return transformed


mat12 = np.array([[1], [0]])
mat13 = np.array([[0], [1]])

result5 = transform_vector(reflection_yaxis, mat12, mat13)




print(f"Original Vector 1 : {mat12}")

print(f"Original Vector 2 : {mat13}")

print(f"Transformed Matrix: {result5}")

"""
Original Vector 1 : 

[[1]
 [0]]
 
Original Vector 2 : 

[[0]
 [1]]
 
Transformed Matrix:
 
[[-1  0]
 [ 0  1]]

"""





# apply a shear transformation


img = cv2.imread(filename='images/leaf_original.png', 0)
plt.imshow(img)


image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(image_rotated)


rows,cols = image_rotated.shape


M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, M, (int(cols), int(rows)))
plt.imshow(image_rotated_sheared)



image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)



M_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
M_shear_x = np.array([[1, 0.5], [0, 1]])




print("90 degrees clockwise rotation matrix:\n", M_rotation_90_clockwise)
print("Matrix for the shear along x-axis:\n", M_shear_x)

"""
90 degrees clockwise rotation matrix:
 [[ 0  1]
 [-1  0]]
Matrix for the shear along x-axis:
 [[1.  0.5]
 [0.  1. ]]

"""




print("M_rotation_90_clockwise by M_shear_x:\n", M_rotation_90_clockwise @ M_shear_x)
print("M_shear_x by M_rotation_90_clockwise:\n", M_shear_x @ M_rotation_90_clockwise)

"""
M_rotation_90_clockwise by M_shear_x:
 [[ 0.   1. ]
 [-1.  -0.5]]
M_shear_x by M_rotation_90_clockwise:
 [[-0.5  1. ]
 [-1.   0. ]]
"""




