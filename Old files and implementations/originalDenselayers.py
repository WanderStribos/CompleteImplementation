import tensorflow as tf
import numpy as np
from tensorflow import keras

#Very old version of the dense layer, which uses tensorflow Dense layers to keep track of the U S and V matrices. Mostly used when the author was still figuring out how Tensorflow worked.
class SVDDense(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):

        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        values = initializer(shape=(input_shape[-1], self.units))

        #It is not possible to directly set weights on creation, so for now we're generating them as zeros and calling setweights which needs numpy. (For some reason)
        u, sigma, vt = np.linalg.svd(values, full_matrices=False) #It is not possible to directly set weights on creation, so for now we're generating them as zerosand calling setweights which requires numpy. (For some reason)

        rank = len(sigma)

        print(input_shape)

        self.u = keras.layers.Dense(rank, use_bias = False)
        self.u.build(input_shape)
        self.u.set_weights(u)
        self.sigma = keras.layers.Dense(rank, use_bias=False)
        self.sigma.build([None, rank])
        self.sigma.set_weights(sigma)
        self.vt = keras.layers.Dense(self.units, use_bias=False)
        self.vt.build([None, self.units])
        self.vt.set_weights(vt)
        
        # self.sig = self.add_weight(
        #     shape=(rank, rank),
        #     initializer="zeros",
        #     trainable=True,
        #     name="sigma"
        # )
        # self.u = self.add_weight(
        #     shape=(self.units, rank),
        #     initializer="zeros",
        #     trainable=True,
        #     name="u"
        # )
        # self.vt = self.add_weight(
        #     shape=(rank, input_shape[-1]),
        #     initializer="zeros",
        #     trainable=True,
        #     name="vt"
        # )
        self.b = self.add_weight(
            shape=(self.units,), initializer=initializer, trainable=True, name="bias"
        )

        # weights = self.get_weights()

        # newWeights = [np.diag(sig), u, vt, b]

        # for i in range(4):
        #     print("Layers:", i, "has shape", np.shape(weights[i]))
        #     print("np:", i, "has shape", np.shape(newEwights[i]))


    def call(self, inputs):
        x = self.u(inputs)
        x = self.sigma(x)
        x = self.vt(x)
        # print("heh")
        # mult = tf.matmul(self.vt, inputs)
        # print("mult", mult)
        # mult = tf.matmul(self.sig, mult)
        # print("mult", mult)
        # mult = tf.matmul(self.u, mult)
        # print("mult", mult)
        # print("huh")
        return x + self.b
    