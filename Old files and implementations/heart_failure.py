import numpy as np
import tensorflow as tf
import NewNoSublayers as layers
import CustomModel
from tensorflow import keras
import os
import pandas as pd
import urllib.request as urllib2 

#An early implementation of the total model, which focussed on the heart_failure dataset from Kaggle. 
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
lam_comp = 1
prune_threshold = 0.9
layercount = 2


# Construct an instance of CustomModel
inputs = keras.Input(shape=(np.shape(x)[-1],))
outputs = layers.SVDDense(
    units = 10,
    lam_comp = lam_comp,
    prune_threshold = prune_threshold,
    layercount = layercount,
    compcount=1
)(inputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
# outputs = layers.SVDDense(
#     units = 5,
#     lam_comp = lam_comp,
#     prune_threshold = prune_threshold,
#     layercount = layercount
# )(outputs)
# outputs = keras.layers.Activation('relu')(outputs)
outputs = layers.SVDDense(
    units = 1,
    lam_comp = lam_comp,
    prune_threshold = prune_threshold,
    layercount = layercount,
    compcount = 1
)(outputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
model = CustomModel.CustomModel(inputs, outputs)

# We don't pass a loss or metrics here.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.5, clipnorm=1.0))

# Just use `fit` as usual -- you can use callbacks, etc.
model.summary()
model.fit(x, y, epochs=20)


#Recompile the model to only use the pruned version.
outputs = inputs
for layerIndex in range(1, len(model.layers)):
    layer = model.layers[layerIndex]
    print(layer)

    if isinstance(layer, layers.SVDDense):
        print(int(layer.rank))
        if (layer.input_shape[-1] * layer.units) > (int(layer.rank) * (layer.input_shape[-1] + layer.units)):
            weights = layer.get_weights()
            outputs = layers.DecomposedDense(
                units = layer.units,
                u_sigma = np.matmul(weights[0][:, :int(layer.rank)], np.diag(weights[1][:int(layer.rank)])),
                vt = weights[2][:int(layer.rank)],
                b = weights[3]
            )(outputs)
        else:
            weights = layer.get_weights()
            outputs = layers.RecomposedDense(
                units = layer.units,
                weights = np.matmul(np.matmul(weights[0][:, :int(layer.rank)], np.diag(weights[1][:int(layer.rank)])), weights[2][:int(layer.rank)]),
                b = weights[3]
            )(outputs)
    else: #if isinstance(layer, keras.activations) :
        new_layer = layer.__class__.from_config(layer.get_config())
        outputs = new_layer(outputs)
    # else:
    #     raise TypeError("unexpected layer type")



newModel = CustomModel.CustomModel(inputs, outputs)
newModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.0, clipnorm=1.0))
newModel.summary()
newModel.fit(x, y, epochs=5)

print(np.array(range(1, len(model.layers))))

# for layerIndex in range(1, len(model.layers)):
#     layer = newModel.layers[layerIndex]
#     print("Next layer")
#     print(layer.rank)
    #print(layer.u[:, :layer.rank])

    # print(tf.matmul(layer.u[:, :layer.rank], layer.u[:, :layer.rank], transpose_a=True) - tf.eye(int(layer.rank)))
    # print(layer.sigma[:layer.rank])
    # print(tf.matmul(layer.vt[:layer.rank], layer.vt[:layer.rank], transpose_b=True) - tf.eye(int(layer.rank)))