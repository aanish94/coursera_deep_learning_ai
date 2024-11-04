import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid


X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear',)
a1 = linear_layer(X_train[0].reshape(1,1))

w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")

set_w = np.array([[200]])
set_b = np.array([100])
# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

prediction_tf = linear_layer(X_train)

###################
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)
model.summary()

logistic_layer = model.get_layer('L1')
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])

a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
