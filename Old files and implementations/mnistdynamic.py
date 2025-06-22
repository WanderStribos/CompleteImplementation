import numpy as np
import tensorflow as tf
import NewNoSublayers as layers
import CustomModel
from tensorflow import keras


#Early non-dynamic implementation of the MNIST dataset before the system was fully working, left here for posterity.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x = x_train
y = y_train

y = keras.utils.to_categorical(y, num_classes=10)

# train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
# train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
# x = np.loadtxt(urllib2.urlopen(train_data_url))
# y = np.loadtxt(urllib2.urlopen(train_resp_url))

#Set hyperparameters:
#lam_struct, q_ort, _u_sort are fixed in the paper, and do not need to be actively set.
lam_comp = 10
prune_threshold = 0.8
layercount = 2


# Construct an instance of CustomModel
inputs = keras.Input(shape=(np.shape(x)[-2], np.shape(x)[-1]))
flattened = keras.layers.Flatten()(inputs)
outputs = layers.SVDDense(
    units = 700,
    lam_comp = lam_comp,
    prune_threshold = prune_threshold,
    layercount = layercount,
    compcount = 20
)(flattened)
outputs = keras.layers.Activation('sigmoid')(outputs)
outputs = layers.SVDDense(
    units = 500,
    lam_comp = lam_comp,
    prune_threshold = prune_threshold,
    layercount = layercount,
    compcount = 5
)(outputs)
outputs = keras.layers.Activation('sigmoid')(outputs)
outputs = keras.layers.Dense(
    units = 10,
)(outputs)
outputs = keras.layers.Activation('softmax')(outputs)
model = CustomModel.CustomModel(inputs, outputs)

# We don't pass a loss or metrics here.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9, clipnorm=1.0))

# Just use `fit` as usual -- you can use callbacks, etc.
model.summary()
model.fit(x, y, epochs=1, batch_size=1000)

#Recompile the model to only use the pruned version.
outputs = flattened
input_size = np.shape(x)[-1]
for layerIndex in range(2, len(model.layers)-2):
    layer = model.layers[layerIndex]

    if isinstance(layer, layers.SVDDense):    
        print(layer.rank)
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

print(np.array(range(1, len(model.layers))))

# for layerIndex in range(1, len(model.layers)):
#     layer = newModel.layers[layerIndex]
#     print("Next layer")
#     print(layer.rank)
    #print(layer.u[:, :layer.rank])

    # print(tf.matmul(layer.u[:, :layer.rank], layer.u[:, :layer.rank], transpose_a=True) - tf.eye(int(layer.rank)))
    # print(layer.sigma[:layer.rank])
    # print(tf.matmul(layer.vt[:layer.rank], layer.vt[:layer.rank], transpose_b=True) - tf.eye(int(layer.rank)))