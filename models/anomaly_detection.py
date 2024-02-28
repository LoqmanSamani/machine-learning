import numpy as np
import matplotlib.pyplot as plt


""" Anomaly Detection with Density Estimation using Gaussian Distribution """


class AnomalyDetection(object):

    def __init__(self, epsilon=1e-5, alpha=1000, not_out=0, out=1):

        self.epsilon = epsilon
        self.alpha = alpha
        self.not_out = not_out
        self.out = out

        self.anomalies = []
        self.anomaly_indices = []
        self.f_score = None
        self.mu_val = None
        self.var_val = None
        self.mu_train = None
        self.var_train = None

    def estimate_gaussian(self, X):

        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        std = np.std(X, axis=0)

        stats = {
            "mu": mu,
            "var": var,
            "std": std
        }

        return stats

    def multivariate_gaussian(self, X, mu, var):
        k = len(mu)

        if var.ndim == 1:
            var = np.diag(var)

        X = X - mu
        p = (2 * np.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) * \
            np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

        return p

    def best_threshold(self, y, y_hat, alpha, not_out, out):

        best_epsilon = 0
        best_f1 = 0

        step_size = (max(y_hat) - min(y_hat)) / alpha

        for epsilon in np.arange(min(y_hat), max(y_hat), step_size):

            outliers = y_hat < epsilon

            tp = np.sum((outliers == out) & (y == out))
            fp = np.sum((outliers == out) & (y == not_out))
            fn = np.sum((outliers == not_out) & (y == out))

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon

        best_params = {
            "epsilon": best_epsilon,
            "f1": best_f1
        }

        return best_params

    def predict(self, X, y_hat, epsilon):

        anomalies = []
        indices = []

        for i in range(len(y_hat)):
            if y_hat[i] < epsilon:
                anomalies.append(list(X[i]))
                indices.append(i)

        return np.array(anomalies), np.array(indices)

    def train(self, X_train, X_val=None, y_val=None):

        if X_val is not None and y_val is not None:

            v_stats = self.estimate_gaussian(
                X=X_val
            )
            self.mu_val = v_stats["mu"]
            self.var_val = v_stats["var"]
            y_hat = self.multivariate_gaussian(
                X=X_val,
                mu=self.mu_val,
                var=self.var_val
            )
            best_params = self.best_threshold(
                y=y_val,
                y_hat=y_hat,
                alpha=self.alpha,
                not_out=self.not_out,
                out=self.out
            )
            self.epsilon = best_params["epsilon"]
            self.f_score = best_params["f1"]
            t_stats = self.estimate_gaussian(
                X=X_train
            )
            self.mu_train = t_stats["mu"]
            self.var_train = t_stats["var"]
            y_hat1 = self.multivariate_gaussian(
                X=X_train,
                mu=self.mu_train,
                var=self.var_train
            )
            anomalies, indices = self.predict(
                X=X_train,
                y_hat=y_hat1,
                epsilon=self.epsilon
            )
            self.anomalies = anomalies
            self.anomaly_indices = indices

        else:
            stats = self.estimate_gaussian(
                X=X_train
            )
            self.mu_train = stats["mu"]
            self.var_train = stats["var"]
            y_hat = self.multivariate_gaussian(
                X=X_train,
                mu=self.mu_train,
                var=self.var_train
            )
            anomalies, indices = self.predict(
                X=X_train,
                y_hat=y_hat,
                epsilon=self.epsilon
            )
            self.anomalies = anomalies
            self.anomaly_indices = indices





X_train = np.load("/home/sam/projects/machine-learning/data/anomaly_detection/X_part1.npy")
X_val = np.load("/home/sam/projects/machine-learning/data/anomaly_detection/X_val_part1.npy")
y_val = np.load("/home/sam/projects/machine-learning/data/anomaly_detection/y_val_part1.npy")

print(X_train[:5, :])
print(X_train.shape)
print(X_val[:5])
print(X_val.shape)
print(y_val[:5])
print(y_val.shape)
"""
[[13.04681517 14.74115241]
 [13.40852019 13.7632696 ]
 [14.19591481 15.85318113]
 [14.91470077 16.17425987]
 [13.57669961 14.04284944]]
(307, 2)
[[15.79025979 14.9210243 ]
 [13.63961877 15.32995521]
 [14.86589943 16.47386514]
 [13.58467605 13.98930611]
 [13.46404167 15.63533011]]
(307, 2)
[0 0 0 0 0]
(307,)
"""

model = AnomalyDetection()
model.train(X_train, X_val=X_val, y_val=y_val)

print(model.epsilon)
print(model.anomalies)
print(model.anomaly_indices)
print(model.f_score)
"""
0.00015729462569735177
[[13.07931049  9.34787812]
 [21.72713402  4.12623222]
 [19.58257277 10.411619  ]
 [23.33986753 16.29887355]
 [18.26118844 17.978309  ]
 [ 4.75261282 24.35040725]]
[300 301 303 304 305 306]
0.8750000000000001
"""

y_hat = model.multivariate_gaussian(
                X=X_val,
                mu=model.mu_val,
                var=model.var_val
            )
anomalies, indices = model.predict(X=X_val, y_hat=y_hat, epsilon=model.epsilon)

X_normal = np.delete(X_train, model.anomaly_indices, axis=0)
X_norm = np.delete(X_val, indices, axis=0)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], color="black")
plt.title("Train Set")
plt.subplot(2, 2, 2)
plt.scatter(X_normal[:, 0], X_normal[:, 1], color="black", label="Normal")
plt.scatter(model.anomalies[:, 0], model.anomalies[:, 1], color="red", label="Anomaly")
plt.title("Anomaly Detection (Train Set)")
plt.legend()
plt.subplot(2, 2, 3)
plt.scatter(X_val[:, 0], X_val[:, 1], color="black")
plt.title("Validation Set")
plt.subplot(2, 2, 4)
plt.scatter(X_norm[:, 0], X_norm[:, 1], color="black", label="Normal")
plt.scatter(anomalies[:, 0], anomalies[:, 1], color="red", label="Anomaly")
plt.title("Anomaly Detection (Validation Set)")
plt.legend()
plt.show()





