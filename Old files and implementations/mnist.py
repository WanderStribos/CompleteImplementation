import numpy as np
import tensorflow as tf
import NoSubLayers as layers
import CustomModel
from tensorflow import keras
import os
import pandas as pd
import urllib.request as urllib2 

#Early non-dynamic implementation of the MNIST dataset before the system was fully working, left here for posterity. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x = x_train
y = y_train

y = keras.utils.to_categorical(y, num_classes=10)

# Construct an instance of CustomModel
inputs = keras.Input(shape=(np.shape(x)[-2], np.shape(x)[-1]))
flattened = keras.layers.Flatten()(inputs)
outputs = keras.layers.Dense(
    units = 500
)(flattened)
outputs = keras.layers.Activation('sigmoid')(outputs)
outputs = keras.layers.Dense(
    units = 250
)(outputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
outputs = keras.layers.Dense(
    units = 10
)(outputs)
outputs = keras.layers.Activation('softmax')(outputs)
#outputs = keras.layers.Activation('sigmoid')(outputs)
model = CustomModel.CustomModel(inputs, outputs)

# We don't pass a loss or metrics here.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9, clipnorm=1.0))

# Just use `fit` as usual -- you can use callbacks, etc.
model.summary()
model.fit(x, y, epochs=20, batch_size=2000)

prune_threshold = 0.6

outputs = flattened
input_size = np.shape(x)[-1]
for layerIndex in range(2, len(model.layers)-2):
    layer = model.layers[layerIndex]
    if isinstance(layer, keras.layers.Dense):

        weights = layer.get_weights()
        
        rank = 0
        u, sigma, vt = np.linalg.svd(weights[0])
        for i in range(len(sigma)):
            if sigma[i] > prune_threshold * sigma[0]:
                rank += 1
            else:
                break
        
        if (input_size * layer.units) > (rank * (input_size + layer.units)):
            outputs = layers.DecomposedDense(
                units = layer.units,
                u_sigma = np.matmul(u[:, :rank], np.diag(sigma[:rank])),
                vt = vt[:rank],
                b = weights[1]
            )(outputs)
        else:
            outputs = layers.RecomposedDense(
                units = layer.units,
                weights = weights[0],
                b = weights[1]
            )(outputs)
        input_size = layer.units
    else:
        new_layer = layer.__class__.from_config(layer.get_config())
        outputs = new_layer(outputs)

layer = model.layers[len(model.layers)-2]
weights = layer.get_weights()
outputs = layers.RecomposedDense(
    units = layer.units,
    weights = weights[0],
    b = weights[1]
    )(outputs)
outputs = keras.layers.Activation('softmax')(outputs)

newModel = CustomModel.CustomModel(inputs, outputs)
newModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.9, clipnorm=1.0))
newModel.summary()
newModel.fit(x, y, epochs=5, batch_size=2000)
