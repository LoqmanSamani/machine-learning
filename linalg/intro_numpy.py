import numpy as np

one_d_array = np.array([12, 23, 68, 79])  # create a one dimensional array
print(one_d_array)


# Create an array with 3 integers, starting from the default integer 0
array1 = np.arange(3)
print(array1)

array2 = np.arange(stop=14, start=4, step=3)
print(array2)


array3 = np.linspace(start=12, stop=100, num=40)
print(array3)

"""
[ 12.          14.25641026  16.51282051  18.76923077  21.02564103
  23.28205128  25.53846154  27.79487179  30.05128205  32.30769231
  34.56410256  36.82051282  39.07692308  41.33333333  43.58974359
  45.84615385  48.1025641   50.35897436  52.61538462  54.87179487
  57.12820513  59.38461538  61.64102564  63.8974359   66.15384615
  68.41025641  70.66666667  72.92307692  75.17948718  77.43589744
  79.69230769  81.94871795  84.20512821  86.46153846  88.71794872
  90.97435897  93.23076923  95.48717949  97.74358974 100.        ]

"""

print(np.round(array3, decimals=2))

"""
[ 12.    14.26  16.51  18.77  21.03  23.28  25.54  27.79  30.05  32.31
  34.56  36.82  39.08  41.33  43.59  45.85  48.1   50.36  52.62  54.87
  57.13  59.38  61.64  63.9   66.15  68.41  70.67  72.92  75.18  77.44
  79.69  81.95  84.21  86.46  88.72  90.97  93.23  95.49  97.74 100.  ]

"""


array4 = np.linspace(start=0, stop=1000, dtype=int, num=20)
print(array4)

"""
[0   52  105  157  210  263  315  368  421  473  526  578  631  684
  736  789  842  894  947 1000]

"""

array5 = np.linspace(start=0, stop=1000, dtype=float, num=20)
print(array5)

"""
[   0.           52.63157895  105.26315789  157.89473684  210.52631579
  263.15789474  315.78947368  368.42105263  421.05263158  473.68421053
  526.31578947  578.94736842  631.57894737  684.21052632  736.84210526
  789.47368421  842.10526316  894.73684211  947.36842105 1000.        ]
"""

array6 = np.array(["welcome!!!!!"])
print(array6)
print(array6.dtype)

"""
['welcome!!!!!']
<U12  # 12 is the length of "welcome!!!!!"
"""


array7 = np.ones(5)
print(array7)

"""
[1. 1. 1. 1. 1.]
"""

array8 = np.zeros(8)
print(array8)


"""
[0. 0. 0. 0. 0. 0. 0. 0.]
"""
# np.empty() creates an array with uninitialized elements from available memory space and may be faster to execute.
array9 = np.empty(10)
print(array9)

"""
[1.13661471e-313 0.00000000e+000 2.21355846e+214 1.69198340e+190
 3.99528230e+252 2.18905908e+232 4.21264844e+228 5.48412867e-322
 0.00000000e+000 0.00000000e+000]
"""

array10 = np.random.rand(10)
print(array10)

"""
[0.25819661 0.0616255  0.00114727 0.84497806 0.6218634  0.91539198
 0.61929428 0.64744195 0.80927495 0.34427897]

"""


# create a two-dimensional array
two_d_array = np.array([[1, 3, 4], [5, 6, 7]])
print(two_d_array)


"""
[[1 3 4]
 [5 6 7]]
"""


array11 = np.array([1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 3, 6])

array12 = np.reshape(array11, newshape=(3, 4))
print(array12)

"""
[[1 2 3 4]
 [5 6 7 7]
 [8 9 3 6]]
"""

print(array12.ndim)
"""
2
"""


print(array12.shape)
"""
(3, 4)
"""

print(array12.size)
"""
12
"""


# elementwise operations

array13 = np.array([1, 2, 3, 4])
array14 = np.array([7, 8, 9, 10])

print(array13 + array14)
"""
[8 10 12 14]
"""

print(array13 - array14)
"""
[-6 -6 -6 -6]
"""

print(array13 * array14)
"""
[ 7 16 27 40]
"""

print(array13 / array14)
"""
[0.14285714   0.25   0.33333333   0.4]
"""

print(array14 * 12)

"""
[ 84  96 108 120]
"""

# Indexing
print(array13[2])
"""
3
"""


print(array12[2][3])

"""
6
"""

# slicing array[start:end:step]

print(array14[1:3])

"""
[8 9]
"""

print(array14[::-1])
"""
[10  9  8  7]
"""
print(array14[::2])

"""
[7 9]
"""


# Note that a == a[:] == a[::]
print(array14[::])
"""
[ 7  8  9 10]
"""

print(array12[:, 2])

"""
[3 7 3]
"""


# stacking is a feature of NumPy that leads to increased customization of arrays.
# It means to join two or more arrays, either horizontally or vertically,
# meaning that it is done along a new axis.

array15 = np.array([
    [1, 2, 3],
    [5, 6, 7],
    [6, 7, 8]
])

array16 = np.array([
    [6, 9, 1],
    [6, 3, 4],
    [1, 6, 0]
])


# Stack the arrays vertically

v_array = np.vstack((array15, array16))
print(v_array)

"""
[[1 2 3]
 [5 6 7]
 [6 7 8]
 [6 9 1]
 [6 3 4]
 [1 6 0]]
"""

# Stack the arrays horizontally

h_array = np.hstack((array15, array16))

print(h_array)

"""
[
 [1 2 3 6 9 1]
 [5 6 7 6 3 4]
 [6 7 8 1 6 0]
]
"""






