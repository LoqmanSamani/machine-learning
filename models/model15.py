import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense, Input, Dot
from tensorflow.keras.activations import linear, relu
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import csv


class ContentBasedFiltering(object):

    def __init__(self, set_seed=True, seed=1):

        self.set_seed = set_seed
        self.seed = seed

    def load_data(self, path):
        data = np.genfromtxt(path, delimiter=",")
        return data

    def normalize_data(self, data, method):

        if method == "min-max normalization":
            scaler = MinMaxScaler()
            scaler.fit(data.reshape(-1, 1))
            normalized_data = scaler.transform(data.reshape(-1, 1))
        else:
            scaler = StandardScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)

        return normalized_data





    def split_data(self, data, train_size=0.8, shuffle=True, random_state=1):

        train_set, test_set = train_test_split(
            data,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state
        )
        return train_set, test_set

    def item_neural_network(self, num_features, num_outputs, units, hidden_activations, output_activation):

        item_nn = Sequential([

            Dense(units=units[0], activation=hidden_activations[0]),
            Dense(units=units[1], activation=hidden_activations[1]),
            Dense(units=num_outputs, activation=output_activation)

        ])

        input_item = Input(shape=num_features)
        output_item = item_nn(input_item)
        output_item = tf.linalg.l2_normalize(output_item, axis=1)

        return input_item, output_item

    def user_neural_network(self, num_features, num_outputs, units, hidden_activations, output_activation):

        user_nn = Sequential([

            Dense(units=units[0], activation=hidden_activations[0]),
            Dense(units=units[1], activation=hidden_activations[1]),
            Dense(units=num_outputs, activation=output_activation)

        ])

        input_user = Input(shape=num_features)
        output_user = user_nn(input_user)
        output_user = tf.linalg.l2_normalize(output_user, axis=1)

        return input_user, output_user

    def combine_neural_networks(self, input_item, output_item, input_user, output_user):

        output = Dot(axes=1)([output_user, output_item])
        model = tf.keras.Model([input_user, input_item], output)

        return model

    def train(self, user_train, item_train, y_train, num_item_features,
              num_user_features, num_outputs, item_units, user_units,
              hidden_activations, output_activation, user_test=None,
              item_test=None, y_test=None, epochs=50, learning_rate=1e-2, evaluate=True):

        if self.set_seed:
            tf.random.set_seed(self.seed)

        input_item, output_item = self.item_neural_network(
            num_features=num_item_features,
            num_outputs=num_outputs,
            units=item_units,
            hidden_activations=hidden_activations,
            output_activation=output_activation
        )

        input_user, output_user = self.user_neural_network(
            num_features=num_user_features,
            num_outputs=num_outputs,
            units=user_units,
            hidden_activations=hidden_activations,
            output_activation=output_activation
        )

        model = self.combine_neural_networks(
            input_item=input_item,
            output_item=output_item,
            input_user=input_user,
            output_user=output_user
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=MeanSquaredError()
        )

        print("***Start Train***")
        model.fit(
            x=[user_train, item_train],
            y=y_train,
            epochs=epochs
        )

        if evaluate:
            print("***Start Evaluation***")
            model.evaluate(
                x=[user_test, item_test],
                y=y_test
            )


user_path = '/home/sam/projects/machine-learning/data/content_based/content_user_train.csv'
item_path = '/home/sam/projects/machine-learning/data/content_based/content_item_train.csv'
target_path = '/home/sam/projects/machine-learning/data/content_based/content_y_train.csv'

model = ContentBasedFiltering()

users = model.load_data(path=user_path)
print(users.shape)
print(users[:5, :])
"""
(50884, 17)
[[ 2.   22.    4.    3.95  4.25  0.    0.    4.    4.12  4.    4.04  0.
   3.    4.    0.    3.88  3.89]
 [ 2.   22.    4.    3.95  4.25  0.    0.    4.    4.12  4.    4.04  0.
   3.    4.    0.    3.88  3.89]
 [ 2.   22.    4.    3.95  4.25  0.    0.    4.    4.12  4.    4.04  0.
   3.    4.    0.    3.88  3.89]
 [ 2.   22.    4.    3.95  4.25  0.    0.    4.    4.12  4.    4.04  0.
   3.    4.    0.    3.88  3.89]
 [ 2.   22.    4.    3.95  4.25  0.    0.    4.    4.12  4.    4.04  0.
   3.    4.    0.    3.88  3.89]]
"""

items = model.load_data(path=item_path)
print(items.shape)
print(items[:5, :])
"""
(50884, 17)
[[6.87400000e+03 2.00300000e+03 3.96183206e+00 1.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00]
 [8.79800000e+03 2.00400000e+03 3.76136364e+00 1.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00]
 [4.69700000e+04 2.00600000e+03 3.25000000e+00 1.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00]
 [4.85160000e+04 2.00600000e+03 4.25233645e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00]
 [5.85590000e+04 2.00800000e+03 4.23825503e+00 1.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00
  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00]]
"""

targets = model.load_data(path=target_path)
print(targets.shape)
print(targets[:20])
"""
(50884,)
[4.  3.5 4.  4.  4.5 5.  4.5 3.  3.  3.  3.  4.  4.  4.  4.  3.  3.  3.
 3.  3. ]

"""

user_scaled = model.normalize_data(data=users, method="standard")
item_scaled = model.normalize_data(data=items, method="standard")
target_scaled = model.normalize_data(data=targets, method="min-max normalization")


user_train, user_test = model.split_data(data=user_scaled, train_size=0.8, shuffle=True, random_state=1)
item_train, item_test = model.split_data(data=item_scaled, train_size=0.8, shuffle=True, random_state=1)
target_train, target_test = model.split_data(data=target_scaled, train_size=0.8, shuffle=True, random_state=1)

print(user_train.shape)
print(item_train.shape)
print(target_train.shape)
"""
(40707, 17)
(40707, 17)
(40707, 1)
"""

print(user_test.shape)
print(item_test.shape)
print(target_test.shape)
"""
(10177, 17)
(10177, 17)
(10177, 1)
"""


model.train(
    user_train=user_train[:, 3:],
    item_train=item_train[:, 1:],
    y_train=target_train,
    num_item_features=item_train.shape[1]-1,
    num_user_features=user_train.shape[1]-3,
    num_outputs=32,
    item_units=[256, 128],
    user_units=[256, 128],
    hidden_activations=[relu, relu],
    output_activation=linear,
    user_test=user_test[:, 3:],
    item_test=item_test[:, 1:],
    y_test=target_test,
    epochs=50,
    learning_rate=1e-2,
    evaluate=True
)

"""
***Start Train***
Epoch 1/50
1273/1273 [==============================] - 3s 2ms/step - loss: 0.0313
Epoch 2/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0281
Epoch 3/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0271
Epoch 4/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0261
Epoch 5/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0254
Epoch 6/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0247
Epoch 7/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0241
Epoch 8/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0237
Epoch 9/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0233
Epoch 10/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0230
Epoch 11/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0225
Epoch 12/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0220
Epoch 13/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0216
Epoch 14/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0211
Epoch 15/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0207
Epoch 16/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0203
Epoch 17/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0200
Epoch 18/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0197
Epoch 19/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0194
Epoch 20/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0192
Epoch 21/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0190
Epoch 22/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0188
Epoch 23/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0186
Epoch 24/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0184
Epoch 25/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0182
Epoch 26/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0180
Epoch 27/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0178
Epoch 28/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0176
Epoch 29/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0175
Epoch 30/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0173
Epoch 31/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0172
Epoch 32/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0171
Epoch 33/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0170
Epoch 34/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0168
Epoch 35/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0167
Epoch 36/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0165
Epoch 37/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0165
Epoch 38/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0164
Epoch 39/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0163
Epoch 40/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0162
Epoch 41/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0161
Epoch 42/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0160
Epoch 43/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0159
Epoch 44/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0159
Epoch 45/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0158
Epoch 46/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0156
Epoch 47/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0156
Epoch 48/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0156
Epoch 49/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0155
Epoch 50/50
1273/1273 [==============================] - 2s 2ms/step - loss: 0.0154
***Start Evaluation***
319/319 [==============================] - 0s 1ms/step - loss: 0.0194

"""







