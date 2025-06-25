import numpy as np
import tensorflow as tf
import CustomModel
import BatchedLayer as layer
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x = x_train
y = y_train

y = keras.utils.to_categorical(y, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


# Construct an instance of CustomModel
inputs = keras.Input(shape=(np.shape(x)[-2], np.shape(x)[-1]))
flattened = keras.layers.Flatten()(inputs)
outputs = layer.SVDDense(
    units = 600,
    lam_comp = 6,
    prune_threshold = 0.8,
    pruning_batch_size = 10,
    mu_ort = 2,
    mu_sing = 0.5,
    name = "first"
)(flattened)
outputs = keras.layers.Activation('relu')(outputs)
outputs = layer.SVDDense(
    units = 300,
    lam_comp = 5,
    prune_threshold = 0.5,
    pruning_batch_size = 5,
    mu_ort = 2,
    mu_sing = 0.5,
    name = "second"
)(outputs)
outputs = keras.layers.Activation('relu')(outputs)
outputs = keras.layers.Dense(
    units = 10,
)(outputs)
outputs = keras.layers.Activation('softmax')(outputs)
model = layer.CustomMetrics.CustomModel(inputs, outputs)


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3, momentum=0.4, clipnorm=6.0))
model.summary()
firstFit = model.fit(x, y, epochs=200, batch_size=1000, validation_data=(x_test, y_test))

#Recompile the model to only use the pruned version.
outputs = flattened
input_size = np.shape(x)[-1]
for layerIndex in range(2, len(model.layers)-2):
    layer = model.layers[layerIndex]

    if isinstance(layer, layer.SVDDense):  
        if (layer.input_shape[-1] * layer.units) > (int(layer.rank) * (layer.input_shape[-1] + layer.units)):
            weights = layer.get_weights()
            outputs = layer.DecomposedDense(
                units = layer.units,
                u_sigma = np.matmul(weights[0][:, :int(layer.rank)], np.diag(weights[1][:int(layer.rank)])),
                vt = weights[2][:int(layer.rank)],
                b = weights[3]
            )(outputs)
            print('Kept decomposed layer', layerIndex, 'with rank: ', layer.rank)
            print(layer.sigma[:layer.rank])
        else:
            weights = layer.get_weights()
            outputs = layer.RecomposedDense(
                units = layer.units,
                weights = np.matmul(np.matmul(weights[0][:, :int(layer.rank)], np.diag(weights[1][:int(layer.rank)])), weights[2][:int(layer.rank)]),
                b = weights[3]
            )(outputs)
            print(layerIndex, 'was too complex and was recomposed. It had rank', layer.rank)
    else: #if isinstance(layer, keras.activations) :
        new_layer = layer.__class__.from_config(layer.get_config())
        outputs = new_layer(outputs)
    # else:
    #     raise TypeError("unexpected layer type")
layer = model.layers[len(model.layers)-2]
weights = layer.get_weights()
outputs = layer.RecomposedDense(
    units = layer.units,
    weights = weights[0],
    b = weights[1]
    )(outputs)
outputs = keras.layers.Activation('softmax')(outputs)


newModel = CustomModel.CustomModel(inputs, outputs)
newModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.6, clipnorm=1.0))
newModel.summary()
newModel.fit(x, y, epochs=10, batch_size=2000, validation_data=(x_test, y_test))

newModel.fit(x_test, y_test, epochs=1, batch_size=60000)
model.fit(x_test, y_test, epochs=1, batch_size=60000)

    
fig, ax1 = plt.subplots(1,1)

ax1.plot(firstFit.history['acc'], color='g', label="Training accuracy")
ax1.plot(firstFit.history['val_acc'], color='b', label="Validation accuracy")

ax2 = ax1.twinx()
ax2.plot(firstFit.history['rank_1'], color='y', label="First layer rank")
ax2.plot(firstFit.history['rank_2'], color='r', label="Second layer rank")
ax2.set_ylim(ymin=0)

#ax[0].plot(firstFit.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
ax1.legend(loc=4, bbox_to_anchor=(0, 0, 1, 1), shadow=True)
ax2.legend(loc=4, bbox_to_anchor=(-0.39, 0, 1, 1), shadow=True)