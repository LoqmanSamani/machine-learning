import tensorflow as tf
import numpy as np
import time
import os


""" Implement collaborative filtering to build a recommender system """


class CollaborativeFiltering(object):
    def __init__(self, Y=None, R=None, W=None, b=None, X=None, print_cost=True):

        self.Y = Y
        self.R = R
        self.W = W
        self.b = b
        self.X = X
        self.print_cost = print_cost
        self.cost_time = []

    def initialize_parameters(self, Y, num_features):

        num_movies, num_users = Y.shape

        W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
        X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
        b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

        return X, W, b

    def compute_cost(self, X, W, b, Y, R, lambda_):

        cost_ = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
        cost = 0.5 * tf.reduce_sum(cost_ ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))

        return cost


    def mean_normalization(self, Y, R):

        Y_mean = tf.reduce_sum(Y * R, axis=1) / (tf.reduce_sum(R, axis=1) + 1e-12)
        Y_mean = tf.reshape(Y_mean, (-1, 1))

        Y_norm = Y - tf.math.multiply(Y_mean, R)

        return Y_mean, Y_norm

    def save_model(self, checkpoint_dir):

        # Ensure the checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = tf.train.Checkpoint(X=self.X, W=self.W, b=self.b)

        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Model saved in {checkpoint_prefix}")

    def restore_model(self, checkpoint_dir):

        checkpoint = tf.train.Checkpoint(X=self.X, W=self.W, b=self.b)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Model restored from {latest_checkpoint}")
        else:
            print("No checkpoint found.")

    def predict(self, X, W, b):
        """
        Predict ratings for users and movies.

        Parameters:
        - X: Movie features matrix (num_movies x num_users).
        - W: User features matrix (num_users x num_features).
        - b: Bias term (1 x num_users).

        Returns:
        - Y_pred: Predicted ratings matrix (num_movies x num_users).
        """
        Y_pred = tf.linalg.matmul(X, tf.transpose(W)) + b

        Y_norm, Y_mean = self.mean_normalization(
            Y=self.Y,
            R=self.R
        )

        predict = Y_pred + Y_mean

        return predict

    def train(self, Y, num_features, epochs=100, learning_rate=1e-2,
              lambda_=1, num=10, path="./checkpoint", save_parameters=False, seed=True):

        if seed:
            tf.random.set_seed(1234)

        Y = tf.constant(Y, dtype=tf.float64)
        R = tf.cast(tf.math.not_equal(Y, 0), dtype=tf.float64)  # create the indicator matrix from Y

        Y_norm, Y_mean = self.mean_normalization(
            Y=Y,
            R=R
        )

        X, W, b = self.initialize_parameters(
            Y=Y_norm,
            num_features=num_features
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        for i in range(epochs):

            start_time = time.time()

            with tf.GradientTape() as tape:

                cost = self.compute_cost(
                    X=X,
                    W=W,
                    b=b,
                    Y=Y_norm,
                    R=R,
                    lambda_=lambda_
                )

            grads = tape.gradient(cost, sources=[X, W, b])  # calculate automatically the gradients
            optimizer.apply_gradients(zip(grads, [X, W, b]))

            stop_time = time.time()
            duration = stop_time - start_time

            cost_time = {
                "Iteration": i,
                "Cost": cost,
                "Duration": duration
            }

            self.cost_time.append(cost_time)

            if self.print_cost:
                if i % num == 0:
                    print(f"Iteration: {i}, Cost: {cost}, Duration: {duration}")

        self.Y = Y
        self.R = R
        self.X = X
        self.W = W
        self.b = b

        if save_parameters:
            self.save_model(checkpoint_dir=path)




# load movie data

path = open("/home/sam/projects/machine-learning/data/collaborative_filtering/small_movies_Y.csv", "rb")
save_path = "/home/sam/projects/machine-learning/data/collaborative_filtering"

data = np.loadtxt(path, delimiter=",")

print(data.shape)
""" (4778, 443) """
print(data[:10, :10])
"""
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [5. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
"""
model = CollaborativeFiltering()

model.train(
    Y=data,
    num_features=20,
    epochs=200,
    learning_rate=1e-1,
    lambda_=2,
    num=10,
    path=save_path,
    save_parameters=True,
    seed=True
)
"""
Iteration: 0, Cost: 639056.0852508154, Duration: 0.190047025680542
Iteration: 10, Cost: 63456.604815933, Duration: 0.11349725723266602
Iteration: 20, Cost: 40549.22230621627, Duration: 0.1031196117401123
Iteration: 30, Cost: 27883.396572564878, Duration: 0.09331965446472168
Iteration: 40, Cost: 19908.47911618868, Duration: 0.10453963279724121
Iteration: 50, Cost: 14476.448932252722, Duration: 0.11893820762634277
Iteration: 60, Cost: 10721.652591692278, Duration: 0.10712051391601562
Iteration: 70, Cost: 8142.562196839812, Duration: 0.10229706764221191
Iteration: 80, Cost: 6362.230548273204, Duration: 0.11339330673217773
Iteration: 90, Cost: 5120.1605659562, Duration: 0.09776449203491211
Iteration: 100, Cost: 4222.553328749569, Duration: 0.10748744010925293
Iteration: 110, Cost: 3549.3487211386278, Duration: 0.09633088111877441
Iteration: 120, Cost: 3028.1369876733393, Duration: 0.10884952545166016
Iteration: 130, Cost: 2614.248362333465, Duration: 0.09345531463623047
Iteration: 140, Cost: 2278.7944474887872, Duration: 0.10679435729980469
Iteration: 150, Cost: 2002.5370531518065, Duration: 0.10808777809143066
Iteration: 160, Cost: 1772.058182042972, Duration: 0.11040234565734863
Iteration: 170, Cost: 1577.7166627929957, Duration: 0.1093740463256836
Iteration: 180, Cost: 1412.5590098890523, Duration: 0.10553121566772461
Iteration: 190, Cost: 1271.1249632446306, Duration: 0.09987521171569824
Model saved in /home/sam/projects/machine-learning/data/collaborative_filtering/ckpt
"""

print(model.cost_time)
"""
[{'Iteration': 0, 
  'Cost': <tf.Tensor: shape=(), dtype=float64, numpy=639056.0852508154>, 
  'Duration': 0.19602012634277344}, 
 {'Iteration': 1, 
 'Cost': <tf.Tensor: shape=(), dtype=float64, numpy=402408.97883966815> ...
"""

model.restore_model(checkpoint_dir=save_path)

""" 
Model restored from /home/sam/projects/machine-learning/data/collaborative_filtering/ckpt-1
"""








