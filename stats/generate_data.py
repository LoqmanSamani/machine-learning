import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.stats import binom
import pandas as pd
import pickle




# uniform distribution


def uniform_generator(lower_bound, upper_bound, num_samp):

    # X = a + (b − a) × random_number
    uniform1 = [lower_bound + (upper_bound - lower_bound) * np.random.random() for _ in range(num_samp)]

    # using numpy library
    uniform2 = np.random.uniform(low=lower_bound, high=upper_bound, size=num_samp)

    return uniform1, uniform2




uniform1, uniform2 = uniform_generator(2, 5, 1000)
uniform11, uniform21 = uniform_generator(4, 8, 1000)
uniform12, uniform22 = uniform_generator(-3, 5, 1000)



plt.hist(uniform1, histtype="step", label='Uniform 1', color='blue')
plt.hist(uniform2, histtype="step", label='Uniform 2', color='orange')
plt.hist(uniform11, histtype="step", label='Uniform 11', color='red')
plt.hist(uniform21, histtype="step", label='Uniform 21', color='purple')
plt.hist(uniform12, histtype="step", label='Uniform 12', color='black')
plt.hist(uniform22, histtype="step", label='Uniform 22', color='yellow')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Uniform Distributions (stats)')
plt.legend()
plt.show()






# Gaussian Distribution

def norm_generator(mean=0, std=1, size=100):

    gaussian1 = std * np.sqrt(2) * erfinv(2 * np.array([np.random.random() for _ in range(size)]) - 1) + mean

    gaussian2 = np.random.normal(loc=mean, scale=std, size=size)

    return gaussian1, gaussian2


gaussian1, gaussian2 = norm_generator(mean=6, std=3, size=1000)
gaussian11, gaussian21 = norm_generator(mean=2, std=5, size=1000)
gaussian12, gaussian22 = norm_generator(mean=4, std=2, size=1000)

plt.hist(gaussian1, histtype="step", label='Uniform 1', color='blue')
plt.hist(gaussian2, histtype="step", label='Uniform 2', color='orange')
plt.hist(gaussian11, histtype="step", label='Uniform 11', color='red')
plt.hist(gaussian21, histtype="step", label='Uniform 21', color='yellow')
plt.hist(gaussian12, histtype="step", label='Uniform 12', color='black')
plt.hist(gaussian22, histtype="step", label='Uniform 22', color='brown')


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Gaussian Distributions (stats)')
plt.legend()
plt.show()







# Binomial Distribution
def binomial_generator(n, p, size):

    ks = [np.random.random() for _ in range(size)]

    binomial = [binom.ppf(k, n, p) for k in ks]

    return binomial


binom1 = binomial_generator(12, 0.4, 1000)
binom2 = binomial_generator(15, 0.5, 1000)
binom3 = binomial_generator(25, 0.8, 1000)

plt.hist(binom1, alpha=0.5, label='Binom1', color='red')
plt.hist(binom2, alpha=0.5, label='Binom2', color='blue')
plt.hist(binom3, alpha=0.5, label='Binom3', color='orange')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Binomial Distributions')
plt.legend()
plt.show()





# Generate data for dog population
np.random.seed(32)




# height(cm)follows a gaussian distribution
_, height = gaussian1, gaussian2 = norm_generator(mean=40, std=4, size=3000)

# weight(kg) follows gaussian distribution
_, weight = norm_generator(mean=20, std=8, size=3000)

# bark_day, representing the number of days (out of 30) that the dog barks, binomial distribution
bark_day = binomial_generator(n=30, p=0.7, size=3000)

# ear_head_ratio, the ratio between the length of the ears and the length of the head uniform distribution
_, ear_head_ratio = uniform_generator(lower_bound=0.1, upper_bound=0.4, num_samp=3000)



dog_data = {
    "Height": height,
    "Weight": weight,
    "Bark Days": bark_day,
    "Ear Head Ratio": ear_head_ratio
}


dog_data1 = pd.DataFrame(dog_data)



print(len(dog_data1))
""" 3000 """
print(dog_data1.head(20))
"""
       Height     Weight  Bark Days  Ear Head Ratio
0   45.810840  14.243828       19.0        0.359870
1   38.476595  22.507534       20.0        0.119201
2   29.705527  22.082535       16.0        0.135106
3   41.706497   9.307472       20.0        0.323798
4   38.318298  26.067042       23.0        0.198297
5   39.006355   7.280854       23.0        0.221073
6   44.706488  17.616744       18.0        0.364416
7   38.349807  16.328696       21.0        0.141814
8   45.714232  38.336859       19.0        0.194522
9   39.935904  28.425205       22.0        0.321888
10  30.938716  15.896151       19.0        0.373937
11  38.159353  23.125361       19.0        0.227134
12  43.679473  25.201221       23.0        0.302099
13  41.857967  25.330133       18.0        0.396918
14  38.194259  19.395963       20.0        0.300933
15  40.270706  -0.044244       21.0        0.312564
16  47.056977  12.558444       16.0        0.233933
17  40.777701  13.550550       20.0        0.126933
18  38.034072  33.297683       19.0        0.231285
19  44.609011  17.436131       21.0        0.176049
"""




# save the data in a csv file format
dog_data1.to_csv("/home/sam/Documents/projects/machine_learning/data/dog_data.csv", index=False)





pre_loaded_df = pd.read_pickle("/home/sam/Documents/machine_learning/stats/3/df_all_breeds.pkl")
df = pd.DataFrame(pre_loaded_df)
df.to_csv(r'file.csv')


