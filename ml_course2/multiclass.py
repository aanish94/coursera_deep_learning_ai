import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        # Dense(4, activation = 'softmax')    # < softmax activation here
        Dense(4, activation = 'linear')  # more numerically stable
    ]
)
model.compile(
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)

# Output is not probababilities, apply softmax function to get
output = model.predict(X_train)
softmax_output = tf.nn.softmax(output).numpy()

# Use to determine most likely category
for i in range(5):
    print( f"{output[i]}, category: {np.argmax(output[i])}")