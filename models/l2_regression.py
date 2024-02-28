import numpy as np



""" Linear Regression Model with L2 Regularization """


class L2LinearRegression(object):

    def __init__(
            self, max_iter=400, params=None, alpha=1e-2,
            beta=1e-2, sigma=1e-8, lam=1, trigger1=1e-8,
            trigger2=1e-12, n=100
    ):

        self.max_iter = max_iter
        self.params = params
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.lam = lam
        self.trigger1 = trigger1
        self.trigger2 = trigger2
        self.n = n

        self.cost_ = []
        self.params_ = []

    def layer_size(self, X, Y):

        m, n = (X.shape[0], Y.shape[0])

        return m, n

    def param_init(self, m, n, beta):

        if not self.params:

            W = np.random.randn(n, m) * beta
            b = np.random.randn(n, 1) * beta

            params = {"W": W, "b": b}
        else:

            params = self.params

        return params

    def compute_cost(self, Y, Z, W, lam):

        m = Y.shape[1]
        n_cost = np.sum(np.square(Z -Y)) / (2 * m)
        reg = np.sum(np.square(W)) * (lam / (2 * m))

        cost = n_cost + reg

        return cost

    def forward_propagation(self, X, W, b):

        Z = np.matmul(W, X) + b

        return Z

    def back_propagation(self, X, Y, Z, W, lam):

        m = X.shape[1]

        dZ = Z - Y
        dW = np.matmul(dZ, X.T) + (lam * W) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        grads = {"dW": dW, "db": db}

        return grads

    def param_update(self, W, b, dW, db, alpha):

        W -= alpha * dW
        b -= alpha * db

        params = {"W": W, "b": b}

        return params

    def predict(self, X, params):

        W = params["W"]
        b = params["b"]

        Z = self.forward_propagation(
            X=X, W=W, b=b
        )

        return Z

    def train(self, X, Y):

        X = np.transpose(X)
        Y = np.transpose(Y)

        m, n = self.layer_size(X=X, Y=Y)

        if not self.params:
            self.param_init(m=m, n=n, beta=self.beta)

        for i in range(self.max_iter):

            Z = self.forward_propagation(
                X=X, W=self.params["W"], b=self.params["b"]
            )
            cost = self.compute_cost(
                Y=Y, Z=Z, W=self.params["W"], lam=self.lam
            )
            grads = self.back_propagation(
                X=X, Y=Y, Z=Z, W=self.params["W"], lam=self.lam
            )
            params = self.param_update(
                W=self.params["W"], b=self.params["b"],
                dW=grads["dW"], db=grads["db"], alpha=self.alpha
            )

            self.cost_.append(cost)
            self.params_.append(params)
            self.params = params

            if i % self.n == 0:
                print(f"Iteration {i}: {cost}")

            if (params["W"] < self.trigger1).all() and (params["b"] < self.trigger1).all() or cost < self.trigger2:

                print(f"after {i} iterations model is converged!")
                print(f"Final Cost: {cost}")
                break

