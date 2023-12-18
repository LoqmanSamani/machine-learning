import numpy as np
import matplotlib.pyplot as plt

mat1 = np.array([[2, 3], [2, 1]])

basis1 = np.array([[1], [0]])
basis2 = np.array([[0], [1]])






def transform_plot(mat, basis1, basis2):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-6, 6))
    ax.set_yticks(np.arange(-6, 6))

    plt.axis([-6, 6, -6, 6])
    plt.quiver([0, 0], [0, 0], [basis1[0], basis2[0]], [basis1[1], basis2[1]], color=color_original, angles='xy', scale_units='xy',
               scale=1)
    plt.plot([0, basis2[0], basis1[0] + basis2[0], basis1[0]],
             [0, basis2[1], basis1[1] + basis2[1], basis1[1]],
             color=color_original)
    v1_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(basis1)])
    ax.text(basis1[0] - 0.2 + v1_sgn[0], basis1[1] - 0.2 + v1_sgn[1], f'$v_1$', fontsize=14, color=color_original)
    v2_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(basis2)])
    ax.text(basis2[0] - 0.2 + v2_sgn[0], basis2[1] - 0.2 + v2_sgn[1], f'$v_2$', fontsize=14, color=color_original)

    v1_transformed = mat @ basis1
    v2_transformed = mat @ basis2

    plt.quiver([0, 0], [0, 0], [v1_transformed[0], v2_transformed[0]], [v1_transformed[1], v2_transformed[1]],
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, v2_transformed[0], v1_transformed[0] + v2_transformed[0], v1_transformed[0]],
             [0, v2_transformed[1], v1_transformed[1] + v2_transformed[1], v1_transformed[1]],
             color=color_transformed)
    v1_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v1_transformed)])
    ax.text(v1_transformed[0] - 0.2 + v1_transformed_sgn[0], v1_transformed[1] - v1_transformed_sgn[1],
            f'$T(v_1)$', fontsize=14, color=color_transformed)
    v2_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v2_transformed)])
    ax.text(v2_transformed[0] - 0.2 + v2_transformed_sgn[0], v2_transformed[1] - v2_transformed_sgn[1],
            f'$T(v_2)$', fontsize=14, color=color_transformed)

    plt.gca().set_aspect("equal")
    plt.show()




transform_plot(mat1, basis1, basis2)






eig_val1, eig_vec1 = np.linalg.eig(mat1)

print(eig_val1)

""" [ 4. -1.] """

print(eig_vec1)
#  The eigenvectors chosen are the normalized ones, so their norms are 1
"""
[[ 0.83205029 -0.70710678]
 [ 0.5547002   0.70710678]]
"""


#  or

eig_mat1 = np.linalg.eig(mat1)

eig_val2 = eig_mat1[0]
eig_vec2 = eig_mat1[1][:, 0]
eig_vec3 = eig_mat1[1][:, 1]

print(eig_mat1)
""" 
EigResult(eigenvalues=array([ 4., -1.]), eigenvectors=array([[ 0.83205029, -0.70710678], [ 0.5547002 ,  0.70710678]])) 
"""

print(eig_val2)  # eigenvalues
""" [ 4. -1.] """

print(eig_vec2)  # first eigenvector
""" [0.83205029 0.5547002 ] """

print(eig_vec3)  # second eigenvector
""" [-0.70710678  0.70710678] """




