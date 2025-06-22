import tensorflow as tf
import numpy as np
from tensorflow import keras


#Old version of the layers class, that used sublayers for the U, sigma, and V layers
class SVDDense(keras.layers.Layer):
    
    def __init__(self, units=32, lam_struct = 1, u_ort = 1000, u_sort = 1, lam_comp = 0.1, prune_threshold = 0.1  , layercount = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.lam_struct = lam_struct
        self.u_ort = u_ort
        self.u_sort = u_sort
        self.lam_comp = lam_comp
        self.prune_threshold = prune_threshold
        self.layercount = layercount
        self.rank = 1
        self.input_shape = 0
    
    # def setHyper(lam_struct, u_ort, q_struct, lam_comp, prune_threshold, layercount):
    #    self.lam_struct = lam_struct
    #    self.u_ort = u_ort
    #    self.q_struct = q_struct
    #    self.lam_comp = lam_comp
    #    self.prune_threshold = prune_threshold
    #    self.layercount = layercount

    def build(self, input_shape):
        self.input_shape = input_shape

        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        values = initializer(shape=(input_shape[-1], self.units))

        #It is not possible to directly set weights on creation, so for now we're generating them as zeros and calling setweights which needs numpy. (For some reason)
        #u, sigma, vt = np.linalg.svd(values, full_matrices=False) #It is not possible to directly set weights on creation, so for now we're generating them as zerosand calling setweights which requires numpy. (For some reason)
        sigma, u, v = tf.linalg.svd(values, full_matrices=False)

        self.rank = int(tf.shape(sigma)[0])
        print(tf.shape(sigma)[0])
        self.u = customDense(self.rank)
        self.u.build(input_shape[-1])
        #self.u.kernel.assign(u)
        self.sigma = diagonal(self.rank)
        self.sigma.build()
        self.sigma.kernel.assign(sigma)
        self.vt = customDense(self.units)
        self.vt.build(self.rank)
        self.vt.kernel.assign(tf.transpose(v))
        self.b = self.add_weight(
            shape=(self.units,), initializer=initializer, trainable=True, name="bias"
        )
        # weights = self.get_weights()

        # newWeights = [np.diag(sig), u, vt, b]

        # for i in range(4):
        #     print("Layers:", i, "has shape", np.shape(weights[i]))
        #     print("np:", i, "has shape", np.shape(newEwights[i]))


    def call(self, inputs):

        #Calculations for the automatic pruning and compression loss
        sigma_abs = tf.abs(self.sigma)
        ratios = tf.math.divide_no_nan(sigma_abs[1:], sigma_abs[-1])

        #Get indices where sigma_i+1 > threshold * sigma_i and take the largest.
        mask = tf.where(ratios > self.prune_threshold)
        if(tf.shape(mask))[0] > 0:
            tau = tf.reduce_max(mask) 
        else: # If there is no value that is large enough, apparently everything after the first rank can be pruned. 
            tau = 1
        
        

        #Temp test, try to immediately prune everything but 1
        # if(self.rank > 1):
        #     u_weights = self.u.get_weights()
        #     sigma_weights = self.sigma.get_weights()
        #     vt_weights = self.vt.get_weights()
        #     self.rank = 1
        #     self.vt = customDense(self.rank)
        #     self.vt.build(self.input_shape) 
        #     self.vt.set_weights([np.transpose(vt_weights)[:-1]])
        #     self.sigma = diagonal(self.rank[:1])
        #     self.sigma.build()
        #     self.sigma.set_weights([sigma_weights])
        #     self.u = customDense(self.units)
        #     self.u.build([self.rank])
        #     self.u.set_weights([u_weights[:-1]])

        #Calculate loss for this layer
        self.add_loss(
            #structural loss
            (
                self.lam_struct*(   
                    self.u_ort / (self.rank**2) * (
                        tf.norm(tf.matmul(self.u.kernel, self.u.kernel, transpose_a=True) - tf.eye(self.rank)) + 
                        tf.norm(tf.matmul(self.vt.kernel, self.vt.kernel, transpose_b=True) - tf.eye(self.rank))
                    ) +
                    self.u_sort * ( 
                        self.singular_sorting_loss(self.sigma.kernel)
                    )
                ) 
                # +
                # #compression loss
                # self.lam_comp * (
                    
                # )
            ) / self.layercount
        )
        x = self.u(inputs)
        x = self.sigma(x)
        x = self.vt(x)
        return x + self.b
    
    def singular_sorting_loss(self, weights):
        difs = weights[:-1] - weights[:1] #The differences of each value and the one after
        difsum = tf.math.divide_no_nan(tf.math.reduce_sum(tf.nn.relu(difs)), #Sum of negative differences
                                       tf.math.reduce_sum(tf.cast(difs < 0, tf.float32))) #divided by the amount of negative differences (with 0 if dividing by 0)
        sumnegs = tf.math.divide_no_nan(tf.math.reduce_sum(tf.nn.relu(-weights)),  #Sum of negative weights
                                        tf.math.reduce_sum(tf.cast(weights < 0, tf.float32))) #divided by the amount of negative weights (with 0 if dividing by 0)
        return difsum + sumnegs


    
class customDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(customDense, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=[input_shape, self.num_outputs],
            initializer = "zeros",
            trainable=True,
            dtype=self.dtype
            )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
  
class diagonal(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(diagonal, self).__init__()
        self.num_outputs = num_outputs

    def build(self):
        self.kernel = self.add_weight(
            shape=[self.num_outputs],
            initializer = "zeros",
            trainable=True,
            )

    def call(self, inputs):
        return tf.matmul(inputs, tf.linalg.diag(self.kernel))