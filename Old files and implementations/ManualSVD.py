import numpy as np
import tensorflow as tf
import NoSubLayers as layers
import CustomModel
from tensorflow import keras
import os
import pandas as pd
import urllib.request as urllib2 

#Early implementation of static SVD, once again on the Kaggle heart failure dataset.
FMT = 'csv'  # TODO: fill in the file format
FILE_NAME = 'heart'  # TODO: fill in the file name of the data set

PATH_TO_DIR = os.getcwd()
PATH_TO_FILE = os.path.join(PATH_TO_DIR, f'{FILE_NAME}.{FMT}')

heart_data = pd.read_csv(PATH_TO_FILE)

data = heart_data.drop('target', axis=1)
labels = heart_data.target

x = data.to_numpy()
y = labels.to_numpy()


# train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
# train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
# x = np.loadtxt(urllib2.urlopen(train_data_url))
# y = np.loadtxt(urllib2.urlopen(train_resp_url))

#Set hyperparameters:
#lam_struct, q_ort, _u_sort are fixed in the paper, and do not need to be actively set.

# Construct an instance of CustomModel
inputs = keras.Input(shape=(np.shape(x)[-1],))
outputs = keras.layers.Dense(
    units = 10
)(inputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
# outputs = keras.layers.Dense(
#     units = 10
# )(outputs)
outputs = keras.layers.Dense(
    units = 1
)(outputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
model = CustomModel.CustomModel(inputs, outputs)

# We don't pass a loss or metrics here.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.5, clipnorm=1.0))

# Just use `fit` as usual -- you can use callbacks, etc.
model.summary()
model.fit(x, y, epochs=20)

prune_threshold = 0.7

outputs = inputs
input_size = np.shape(x)[-1]
for layerIndex in range(1, len(model.layers)):
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

newModel = CustomModel.CustomModel(inputs, outputs)
newModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.0, clipnorm=1.0))
newModel.summary()
newModel.fit(x, y, epochs=10)
