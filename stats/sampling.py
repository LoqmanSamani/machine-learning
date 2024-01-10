import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


# Central Limit Theorem

# Normal Population
mu = 5
std = 10
N = 100000

gaussian_pop = np.random.normal(mu, std, N)

sns.histplot(gaussian_pop)
plt.xlabel("Variable")
plt.ylabel("Frequency")
plt.title("Gaussian Population")
plt.show()


# sampling
def get_sample(pop, size):

    samples = []
    means = []
    stds = []
    for i in range(len(size)):

        sample = np.random.choice(pop, size=size[i])
        samples.append(sample)
        means.append(np.mean(sample))
        stds.append(np.std(sample))

    return samples, means, stds


sam1, mean1, std1 = get_sample(pop=gaussian_pop, size=[i*7 for i in range(1, 20)])

print(sam1)
print(mean1)
print(std1)

print(np.mean(mean1))
print(np.std(std1))
"""
samples mean = 4.776637570664203   pop mean = 5
samples std = 1.027839467816976   pop std = 10
"""


def get_sample2(pop, size, num_iter):

    samples = []
    means = []
    for i in range(num_iter):
        sample = np.random.choice(pop, size=size)
        samples.append(sample)
        means.append(np.mean(sample))

    return samples, means


sam2, mean2 = get_sample2(gaussian_pop, 30, 1000)

sns.histplot(sam2[0])
plt.show()

sns.histplot(mean2)
plt.show()


sam3, mean3 = get_sample2(gaussian_pop, 200, 10000)

sns.histplot(sam3[0])
plt.show()

sns.histplot(mean3)
plt.show()

# Create the QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(np.array(mean3), plot=ax, fit=True)
plt.show()


# Binomial population

binom_pop = np.random.binomial(n=12, p=.7, size=100000)

sns.histplot(binom_pop)
plt.title("Binomial Distribution")
plt.show()
