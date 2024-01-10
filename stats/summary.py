import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)

path1 = ".../anscombe.csv"

with open(path1, "r") as file:
    data1 = pd.read_csv(file)

print(data1.columns)
""" Index(['x', 'y', 'group'], dtype='object' """
print(len(data1))
""" 44 """
print(data1.head())
"""
   x     y  group
0  10.0  8.04      1
1   8.0  6.95      1
2  13.0  7.58      1
3   9.0  8.81      1
4  11.0  8.33      1

"""

print(data1["group"].nunique())
""" 4 """

print(list(data1.group))
""" [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] 
"""

print(data1.groupby("group").describe())
"""
         x                                               y            \
      count mean       std  min  25%  50%   75%   max count      mean   
group                                                                   
1      11.0  9.0  3.316625  4.0  6.5  9.0  11.5  14.0  11.0  7.500909   
2      11.0  9.0  3.316625  4.0  6.5  9.0  11.5  14.0  11.0  7.500909   
3      11.0  9.0  3.316625  4.0  6.5  9.0  11.5  14.0  11.0  7.500000   
4      11.0  9.0  3.316625  8.0  8.0  8.0   8.0  19.0  11.0  7.500909   

                                                 
            std   min    25%   50%   75%    max  
group                                            
1      2.031568  4.26  6.315  7.58  8.57  10.84  
2      2.031657  3.10  6.695  8.14  8.95   9.26  
3      2.030424  5.39  6.250  7.11  7.98  12.74  
4      2.030579  5.25  6.170  7.04  8.19  12.50  
"""

print(data1.groupby("group").corr())
"""
group                      
1     x  1.000000  0.816421
      y  0.816421  1.000000
2     x  1.000000  0.816237
      y  0.816237  1.000000
3     x  1.000000  0.816287
      y  0.816287  1.000000
4     x  1.000000  0.816521
      y  0.816521  1.000000
"""


plt.scatter(data1["x"], data1["y"])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Anscombe's Quartet")
plt.show()

sub1 = data1[data1["group"] == 1]
sub2 = data1[data1["group"] == 2]
sub3 = data1[data1["group"] == 3]
sub4 = data1[data1["group"] == 4]


plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.scatter(x=sub1.x, y=sub1.y)
plt.xlim(0, 20)
plt.ylim(0, 15)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Group 1")

plt.subplot(2, 2, 2)
plt.scatter(x=sub2.x, y=sub2.y)
plt.xlim(0, 20)
plt.ylim(0, 15)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Group 2")

plt.subplot(2, 2, 3)
plt.scatter(x=sub3.x, y=sub3.y)
plt.xlim(0, 20)
plt.ylim(0, 15)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Group 3")

plt.subplot(2, 2, 4)
plt.scatter(x=sub4.x, y=sub4.y)
plt.xlim(0, 20)
plt.ylim(0, 15)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Group 4")

plt.tight_layout()
plt.show()


path2 = ".../datasaurus.csv"

with open(path2, "r") as file1:
    data2 = pd.read_csv(file1)

print(data2)
"""
        group          x          y
0           dino  55.384600  97.179500
1           dino  51.538500  96.025600
2           dino  46.153800  94.487200
3           dino  42.820500  91.410300
4           dino  40.769200  88.333300
...          ...        ...        ...
1841  wide_lines  33.674442  26.090490
1842  wide_lines  75.627255  37.128752
1843  wide_lines  40.610125  89.136240
1844  wide_lines  39.114366  96.481751
1845  wide_lines  34.583829  89.588902
"""

data21 = data2.groupby("group")

print(data21.describe())
"""
                x                                                         \
            count       mean        std        min        25%        50%   
group                                                                      
away        142.0  54.266100  16.769825  15.560750  39.724115  53.340296   
bullseye    142.0  54.268730  16.769239  19.288205  41.627968  53.842088   
circle      142.0  54.267320  16.760013  21.863581  43.379116  54.023213   
dino        142.0  54.263273  16.765142  22.307700  44.102600  53.333300   
dots        142.0  54.260303  16.767735  25.443526  50.359707  50.976768   
h_lines     142.0  54.261442  16.765898  22.003709  42.293828  53.069678   
high_lines  142.0  54.268805  16.766704  17.893499  41.535981  54.168689   
slant_down  142.0  54.267849  16.766759  18.109472  42.890931  53.135159   
slant_up    142.0  54.265882  16.768853  20.209778  42.810866  54.261345   
star        142.0  54.267341  16.768959  27.024603  41.034210  56.534732   
v_lines     142.0  54.269927  16.769959  30.449654  49.964506  50.362890   
wide_lines  142.0  54.266916  16.770000  27.439632  35.522449  64.550226   
x_shape     142.0  54.260150  16.769958  31.106867  40.091656  47.136458   

                                      y                                   \
                  75%        max  count       mean        std        min   
group                                                                      
away        69.146597  91.639961  142.0  47.834721  26.939743   0.015119   
bullseye    64.798900  91.735539  142.0  47.830823  26.935727   9.691547   
circle      64.972672  85.664761  142.0  47.837717  26.930036  16.326546   
dino        64.743600  98.205100  142.0  47.832253  26.935403   2.948700   
dots        75.197363  77.954435  142.0  47.839829  26.930192  15.771892   
h_lines     66.768274  98.288123  142.0  47.830252  26.939876  10.463915   
high_lines  63.952667  96.080519  142.0  47.835450  26.939998  14.913962   
slant_down  64.469989  95.593416  142.0  47.835896  26.936105   0.303872   
slant_up    64.488010  95.260528  142.0  47.831496  26.938608   5.645777   
star        68.711493  86.435897  142.0  47.839545  26.930275  14.365590   
v_lines     69.504068  89.504851  142.0  47.836988  26.937684   2.734760   
wide_lines  67.453672  77.915874  142.0  47.831602  26.937902   0.217006   
x_shape     71.856923  85.446186  142.0  47.839717  26.930002   4.577661   

                                                        
                  25%        50%        75%        max  
group                                                   
away        24.625892  47.535269  71.803148  97.475771  
bullseye    26.244735  47.382937  72.532852  85.876229  
circle      18.349610  51.025022  77.782382  85.578134  
dino        25.288450  46.025600  68.525675  99.487200  
dots        17.107141  51.299291  82.881589  94.249328  
h_lines     30.479911  50.473527  70.349471  90.458936  
high_lines  22.920843  32.499203  75.940022  87.152208  
slant_down  27.840858  46.401314  68.439430  99.644179  
slant_up    24.756248  45.292238  70.855844  99.579591  
star        20.374135  50.110554  63.548584  92.214989  
v_lines     22.752884  47.113616  65.845391  99.694680  
wide_lines  24.346941  46.279331  67.568127  99.283764  
x_shape     23.470809  39.876211  73.609634  97.837615  
"""

print(set(list(data2.group)))
""" {'away', 'v_lines', 'dino', 'h_lines', 'high_lines', 'star', 'circle', 'bullseye', 'x_shape', 'wide_lines', 'slant_down', 'dots', 'slant_up'} """

subsets = list(set(list(data2.group)))

sub21 = data2[data2.group == subsets[0]]
sub22 = data2[data2.group == subsets[1]]
sub23 = data2[data2.group == subsets[2]]
sub24 = data2[data2.group == subsets[3]]
sub25 = data2[data2.group == subsets[4]]
sub26 = data2[data2.group == subsets[5]]
sub27 = data2[data2.group == subsets[6]]
sub28 = data2[data2.group == subsets[7]]
sub29 = data2[data2.group == subsets[8]]
sub210 = data2[data2.group == subsets[9]]
sub211 = data2[data2.group == subsets[10]]
sub212 = data2[data2.group == subsets[11]]
sub213 = data2[data2.group == subsets[12]]

colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink',
          'brown', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'teal']



plt.figure(figsize=(20, 10))

plt.subplot(4, 4, 1)
plt.scatter(x=sub21.x, y=sub21.y, color=colors[0])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("away")

plt.subplot(4, 4, 2)
plt.scatter(x=sub22.x, y=sub22.y, color=colors[1])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("v_lines")

plt.subplot(4, 4, 3)
plt.scatter(x=sub23.x, y=sub23.y, color=colors[2])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("dino")

plt.subplot(4, 4, 4)
plt.scatter(x=sub24.x, y=sub24.y, color=colors[3])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("h_lines")


plt.subplot(4, 4, 5)
plt.scatter(x=sub25.x, y=sub25.y, color=colors[4])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("star")

plt.subplot(4, 4, 6)
plt.scatter(x=sub26.x, y=sub26.y, color=colors[5])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("circle")

plt.subplot(4, 4, 7)
plt.scatter(x=sub27.x, y=sub27.y, color=colors[6])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("bullseye")

plt.subplot(4, 4, 8)
plt.scatter(x=sub28.x, y=sub28.y, color=colors[7])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("x_shape")

plt.subplot(4, 4, 9)
plt.scatter(x=sub29.x, y=sub29.y, color=colors[8])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("wide_lines")

plt.subplot(4, 4, 10)
plt.scatter(x=sub210.x, y=sub210.y,  color=colors[9])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("slant_down")

plt.subplot(4, 4, 11)
plt.scatter(x=sub211.x, y=sub211.y,  color=colors[10])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Group 4")

plt.subplot(4, 4, 12)
plt.scatter(x=sub212.x, y=sub212.y,  color=colors[11])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("dots")

plt.subplot2grid((4, 4), (3, 1), colspan=2, rowspan=1)
plt.scatter(x=sub213.x, y=sub213.y,  color=colors[12])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("slant_up")


plt.tight_layout()
plt.show()






