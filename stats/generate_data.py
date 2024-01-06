import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.stats import binom





# uniform distribution


def uniform_generator(lower_bound, upper_bound, num_samp):

    # X = a + (b − a) × random_number
    uniform1 = [lower_bound + (upper_bound - lower_bound) * np.random.random() for _ in range(num_samp)]

    # using numpy library
    uniform2 = np.random.uniform(low=lower_bound, high=upper_bound, size=num_samp)

    return uniform1, uniform2




uniform1, uniform2 = uniform_generator(2, 5, 1000)

plt.hist(uniform1, histtype="step", label='Uniform 1', color='blue')
plt.hist(uniform2, histtype="step", label='Uniform 2', color='orange')

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

plt.hist(gaussian1, histtype="step", label='Uniform 1', color='blue')
plt.hist(gaussian2, histtype="step", label='Uniform 2', color='orange')

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







