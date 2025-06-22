import tensorflow as tf
import numpy as np
from tensorflow import keras


#Final version of the custom layers. Features batched compression and pruning.


class SVDDense(keras.layers.Layer):
    
    def __init__(self, units=32, mu_ort = 1000, mu_sing = 1, mu_comp = 0.1, prune_threshold = 0.1  , pruning_batch_size = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.mu_ort = mu_ort
        self.mu_sing = mu_sing
        self.mu_comp = mu_comp
        self.prune_threshold = prune_threshold
        self.rank = 1 #Is replaced later.
        self.pruning_batch_size = pruning_batch_size

    def build(self, input_shape):

        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        values = initializer(shape=(input_shape[-1], self.units))

        #Initialisation of the weight matrices
        #Instead of generating new three different instances of tf.constant_initializer, we initialise the layers as zeros and assign them manually. 
        sigma, u, v = tf.linalg.svd(values, full_matrices=False)
        vt = tf.transpose(v)
        self.rank = int(tf.shape(sigma)[0])

        
        self.u = self.add_weight(
            shape=(input_shape[-1], self.rank), initializer='zeros', trainable=True, name="u"
        )
        self.u.assign(u)

        self.sigma = self.add_weight(
            shape=(self.rank,), initializer='zeros', trainable=True, name="sigma"
        )
        self.sigma.assign(sigma)

        self.vt = self.add_weight(
            shape=(self.rank, self.units), initializer='zeros', trainable=True, name="vt"
        )
        self.vt.assign(vt)

        self.b = self.add_weight(
            shape=(self.units,), initializer=initializer, trainable=True, name="bias"
        )

        #Change the rank to a tensorflow tensor, to allow it to be used during call. Done here as self.addweight requires integers
        self.rank = tf.Variable(self.rank, trainable=False, dtype=tf.int32)


    def call(self, inputs):

        #Truncate matrices. Sadly, tensorflow does not allow for weights to decrease in size without recompiling the model, so for this implementation we actively truncate it. This does mean it only trains the truncated part, but also means the parameters are still saved.
        current_u = self.u[:, :self.rank]
        current_sigma = self.sigma[:self.rank]
        current_vt = self.vt[:self.rank]
        #rank_f = tf.cast(self.rank, tf.float32)

        #Pick the last n singular values, and the last n before that.
        smallest_batch = tf.reduce_sum(current_sigma[-self.pruning_batch_size:])
        secondSmallest_batch = tf.reduce_sum(current_sigma[-self.pruning_batch_size*2:-self.pruning_batch_size])

        #If the last n values are small enough, decrease rank by n. Cannot be below zero!
        tf.cond(
            self.rank > self.pruning_batch_size,
            lambda: self.rank.assign(tf.cond(
                smallest_batch < self.prune_threshold * secondSmallest_batch,
                    lambda: self.rank - self.pruning_batch_size,
                    lambda: self.rank
            )),
            lambda: self.rank
        )
        
        #Calculate structural and compression loss for this layer
        self.add_loss(
            
            (
            #orthonormality loss
            self.mu_ort * (
                tf.reduce_mean(tf.square(tf.matmul(current_u, current_u, transpose_a=True) - tf.eye(tf.shape(current_sigma)[0]))) + 
                tf.reduce_mean(tf.square(tf.matmul(current_vt, current_vt, transpose_b=True) - tf.eye(tf.shape(current_sigma)[0])))
            ) 
            +

            #sorting loss
            self.mu_sing * (
                self.singular_sorting_loss(current_sigma)
            )

            +

            # Compression loss 
            # Different from the original paper, it simply attempts to further decrease the size of the final (smallest) singular value.
            self.mu_comp * smallest_batch
            )
        )

        #Calculate and return forward pass
        x = tf.matmul(inputs, current_u)
        x = tf.matmul(x, tf.linalg.diag(current_sigma))
        x = tf.matmul(x, current_vt)
        return x + self.b
    
    def singular_sorting_loss(self, weights):
        difs = weights[1:] - weights[:-1] #The differences of each value and the one after
        difsum = tf.math.divide_no_nan(tf.reduce_sum(tf.nn.relu(-difs)), #Sum of negative differences
                                       tf.reduce_sum(tf.cast(difs < 0, tf.float32))) #divided by the amount of negative differences (with 0 if dividing by 0)
        sumnegs = tf.math.divide_no_nan(tf.reduce_sum(tf.nn.relu(-weights)),  #Sum of negative weights
                                        tf.reduce_sum(tf.cast(weights < 0, tf.float32))) #divided by the amount of negative weights (with 0 if dividing by 0)
        return difsum + sumnegs

#Used after recompiling. Equal to a normal Dense layer, but with set starting weights.  
class RecomposedDense(keras.layers.Layer):
    def __init__(self, units=32, weights = [], b = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.w = weights
        self.b = b

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(self.w), trainable=True, name="weights"
        )
        
        self.b = self.add_weight(
            shape=(self.units,), initializer=tf.constant_initializer(self.b), trainable=True, name="bias"
        )
        
    def call(self, inputs):
        x = tf.matmul(inputs, self.w)
        return x + self.b
    
#Used after recompiling. Functions as a normal Dense layer, but with the weights matrix decomposed into U*sigma and Vt.    
class DecomposedDense(keras.layers.Layer):
    def __init__(self, units=32, u_sigma = [], vt = [], b = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.rank = 1 #Overwritten during build.
        self.u_sigma = u_sigma
        self.vt = vt
        self.b = b

    def build(self, input_shape):
        
        self.rank = int(tf.shape(self.u_sigma)[1])

        self.u_sigma = self.add_weight(
            shape=(input_shape[-1], self.rank), initializer=tf.constant_initializer(self.u_sigma), trainable=True, name="u_sigma"
        )

        if len(self.vt) == 1:
           self.vt = self.vt[0][0]

        self.vt = self.add_weight(
            shape=(self.rank, self.units), initializer=tf.constant_initializer(self.vt), trainable=True, name="vt"
        )

        self.b = self.add_weight(
            shape=(self.units,), initializer=tf.constant_initializer(self.b), trainable=True, name="bias"
        )

        self.rank = tf.Variable(self.rank, trainable=False, dtype=tf.int32)

        
    def call(self, inputs):

        x = tf.matmul(inputs, self.u_sigma)
        x = tf.matmul(x, self.vt)
        return x + self.b