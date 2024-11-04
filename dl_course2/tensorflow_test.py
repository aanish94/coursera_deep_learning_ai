import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
# optimizer = tf.keras.optimizers.Adam(0.1)


optimizer = tf.keras.optimizers.Adam

def training(x, w, optimizer):

    def cost_fn():
        return x[0] * w ** 2 + x[1] * w + x[2]
    
    for i in range(1000):
        optimizer.minimize(cost_fn, [w])
    return w

w = training(x, w, optimizer)
print(w)
