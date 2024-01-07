import numpy as np
import pandas as pd
import math
from scipy.special import comb


# Naive Bayes Classifier (Continuous)

class EstimateParams(object):

    def __init__(self):
        self.params = {}


    def gaussian_params(self, X):

        mu = round(np.mean(X), 2)
        sigma = round(np.std(X), 2)

        return mu, sigma


    def uniform_params(self, X):

        lower_bound = round(np.min(X), 2)
        upper_bound = round(np.max(X), 2)

        return lower_bound, upper_bound


    def binom_params(self, X):

        n = 30
        p = round(np.mean(X) / 30, 2)

        return n, p


    def call_func(self, X, dist):

        if dist == "Gaussian":
            p1, p2 = self.gaussian_params(X)

        elif dist == "Uniform":
            p1, p2 = self.uniform_params(X)

        elif dist == "Binomial":
            p1, p2 = self.binom_params(X)

        else:
            print(f"There is no distribution named {dist} in this model!")

        return p1, p2



    def train_params(self, data, dists):

        params = {}
        breeds = data.groupby("breed")
        columns = data.columns

        for group, breed in breeds:
            dict1 = {}
            for i in range(len(dists)):

                dist = dists[i]
                p1, p2 = self.call_func(breed[columns[i]], dist)

                if dist == "Gaussian":
                    dict1[columns[i]] = {"mu": p1, "sigma": p2}

                elif dist == "Uniform":
                    dict1[columns[i]] = {"a": p1, "b": p2}

                elif dist == "Binomial":
                    dict1[columns[i]] = {"n": p1, "p": p2}

                else:
                    print(f"There is no distribution named {dist} in this model!")

            params[group] = dict1

        self.params = params









class Prediction(object):



    def pdf_uniform(self, X, lower_bound, upper_bound):

        """Calculates the probability density function (PDF) of a uniform distribution"""

        if lower_bound <= X <= upper_bound:
            pdf = 1 / (upper_bound - lower_bound)
        else:
            pdf = 0

        return pdf



    def pdf_gaussian(self, X, mu, sigma):

        """Calculate the probability density function (PDF) of a Gaussian distribution"""

        pdf = (1 / (sigma * np.sqrt(2 * math.pi))) * np.exp((-1 / 2) * ((X - mu) / sigma) ** 2)

        return pdf





    def pmf_binomial(self, X, size, p):

        """Calculate the probability mass function (PMF) of a binomial distribution(discrete)"""

        pmf = comb(size, X) * (p ** X) * ((1 - p) ** (size - X))

        return pmf





    def class_probs(self, data):

        """Computes the estimated probabilities of each breed"""
        probs = {}
        breeds = data.groupby("breed")

        for group, breed in breeds:
            prob = len(breed) / len(data)
            probs[group] = round(prob, 2)

        return probs




    def compute_prob(self, X, feats, breed, params):

        feat_x = list(zip(feats, X))
        pdf1 = self.pdf_gaussian(X=feat_x[0][1], mu=params[breed]["height"]["mu"], sigma=params[breed]["height"]["sigma"])
        pdf2 = self.pdf_gaussian(X=feat_x[1][1], mu=params[breed]["weight"]["mu"], sigma=params[breed]["weight"]["sigma"])
        pmf = self.pmf_binomial(X=feat_x[2][1], size=params[breed]["bark_days"]["n"], p=params[breed]["bark_days"]["p"])
        pdf3 = self.pdf_uniform(X=feat_x[3][1], lower_bound=params[breed]["ear_head_ratio"]["a"], upper_bound=params[breed]["ear_head_ratio"]["b"])

        prob = pdf1 * pdf2 * pmf * pdf3

        return prob





    def predict(self, X, feats, params, probs):

        breeds = [0.0 for _ in range(len(probs.keys()))]

        for i in range(len(probs.keys())):
            breeds[i] = self.compute_prob(X, feats, i, params) * probs[i]

        prediction = np.argmax(breeds)

        return prediction








file_path = '/home/sam/Documents/projects/machine_learning/data/dog_data1.csv'


with open(file_path, 'rb') as file:
    dog_data = pd.read_csv(file)


df_train = dog_data[:3000]
df_test = dog_data[3000:]

print(len(df_train))

print(len(df_test))
"""
3000
450
"""
print(df_train.head(10))
"""
  Unnamed: 0     height     weight  bark_days  ear_head_ratio  breed
0        2836  39.697810  31.740980        9.0        0.193120      2
1        1002  36.710641  21.140427       26.0        0.163527      0
2        1075  34.726930  19.817954       24.0        0.386113      0
3        1583  32.324884  30.812210       18.0        0.463242      1
4         248  37.691499  21.794333       28.0        0.118190      0
5         814  36.688852  21.125901       26.0        0.165052      0
6        1407  30.844078  27.110196       16.0        0.399051      1
7        3376  38.616784  30.814387        8.0        0.169269      2
8        2700  44.655532  35.990456       12.0        0.281653      2
9         533  35.209095  20.139397       24.0        0.322284      0
"""
print(df_test.head(10))
"""
      Unnamed: 0     height     weight  bark_days  ear_head_ratio  breed
3000        3288  39.463930  31.540511        9.0        0.187827      2
3001        2239  29.038181  22.595451       14.0        0.294587      1
3002        3367  39.104247  31.232212        8.0        0.179801      2
3003        1265  30.214457  25.536142       15.0        0.362809      1
3004         483  35.488089  20.325393       25.0        0.286221      0
3005        1887  28.834193  22.085482       13.0        0.283994      1
3006        2963  40.647968  32.555401        9.0        0.214688      2
3007        1219  28.900405  22.251013       13.0        0.287369      1
3008         277  35.577835  20.385223       25.0        0.275018      0
3009         988  33.971216  19.314144       23.0        0.476799      0
"""


train_data = df_train[["height", "weight", "bark_days", "ear_head_ratio", "breed"]]

parameters = EstimateParams()

parameters.train_params(data=train_data, dists=["Gaussian", "Gaussian", "Binomial", "Uniform"])

print(parameters.params)
"""
{
 0: {'height': {'mu': 35.0, 'sigma': 1.51},
     'weight': {'mu': 20.0, 'sigma': 1.01},
     'bark_days': {'n': 30, 'p': 0.8}, 
     'ear_head_ratio': {'a': 0.1, 'b': 0.6}}, 
 1: {'height': {'mu': 30.01, 'sigma': 2.04}, 
     'weight': {'mu': 25.03, 'sigma': 5.11}, 
     'bark_days': {'n': 30, 'p': 0.5}, 
     'ear_head_ratio': {'a': 0.2, 'b': 0.5}}, 
 2: {'height': {'mu': 39.9, 'sigma': 3.54}, 
     'weight': {'mu': 31.91, 'sigma': 3.04}, 
     'bark_days': {'n': 30, 'p': 0.3}, 
     'ear_head_ratio': {'a': 0.1, 'b': 0.3}}
}

"""

prediction = Prediction()

probs = prediction.class_probs(dog_data[["height", "weight", "bark_days", "ear_head_ratio", "breed"]])
print(probs)
""" {0: 0.35, 1: 0.39, 2: 0.26} """

test_data = df_test[["height", "weight", "bark_days", "ear_head_ratio"]]
feats1 = ["height", "weight", "bark_days", "ear_head_ratio"]
test_labels = df_test[["breed"]]

Y = test_labels.values.tolist()

preds = []

for x in test_data.values.tolist():
    pred = prediction.predict(x, feats1, parameters.params, probs)
    preds.append(pred)

print(preds[:10])
""" [2, 1, 2, 1, 0, 1, 2, 1, 0, 0]"""

accuracy = sum([1 if x == y else 0 for x, y in zip(preds, Y)]) / len(preds)
print(accuracy)
""" 1.0 """
"""
The model achieved 100% accuracy on the test data :)
"""


