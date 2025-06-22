import tensorflow as tf
import numpy as np
from tensorflow import keras

#Old version of SVDDense, only compresses and prunes a single value.
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
        self.rank = 1 #Is replaced later.
        self.input_shape = 0

    def build(self, input_shape):
        self.input_shape = input_shape

        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        values = initializer(shape=(input_shape[-1], self.units))

        #It is not possible to directly set weights on creation, so for now we're generating them as zeros and calling assign
        #Later note: This is not true, but it's a deprecated version.
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

        self.rank = tf.Variable(self.rank, trainable=False, dtype=tf.int32)


    def call(self, inputs):

        current_u = self.u[:, :self.rank]
        current_sigma = self.sigma[:self.rank]
        current_vt = self.vt[:self.rank]
        #rank_f = tf.cast(self.rank, tf.float32)

        tf.cond(
            self.rank > 1,
            lambda: self.rank.assign(tf.cond(
                current_sigma[-1] < self.prune_threshold * current_sigma[-2],
                    lambda: self.rank - 1,
                    lambda: self.rank
            )),
            lambda: self.rank
        )

        #can_decrease = current_sigma[-1] < self.prune_threshold * sigma[-2]
        #sigma_mask = tf.cast(tf.range(tf.shape(self.sigma)[0]) == (self.rank - 1), tf.float32)
        rank_int = tf.cast(self.rank, tf.int32)
        last_sigma = tf.gather(self.sigma, rank_int - 1)#self.sigma[-1]
        
        #Calculate loss for this layer
        self.add_loss(
            
            (
            #structural loss
            self.lam_struct*(   
                #orthogonality
                self.u_ort / (tf.cast(self.rank, tf.float32)**2) * (
                    tf.reduce_mean(tf.square(tf.matmul(current_u, current_u, transpose_a=True) - tf.eye(tf.shape(current_sigma)[0]))) + 
                    tf.reduce_mean(tf.square(tf.matmul(current_vt, current_vt, transpose_b=True) - tf.eye(tf.shape(current_sigma)[0])))
                    # tf.norm(tf.matmul(current_u, current_u, transpose_a=True) - tf.eye(tf.shape(current_sigma)[0])) + 
                    # tf.norm(tf.matmul(current_vt, current_vt, transpose_b=True) - tf.eye(tf.shape(current_sigma)[0]))
                ) 
                +

                #sorting
                self.u_sort * ( 
                    self.singular_sorting_loss(current_sigma)
                )
            ) +

            # Compression loss 
            # Different from the original paper, it simply attempts to further decrease the size of the final (smallest) singular value.
            self.lam_comp * last_sigma / self.layercount
            ) / self.layercount
        )
        x = tf.matmul(inputs, current_u)
        x = tf.matmul(x, tf.linalg.diag(current_sigma))
        x = tf.matmul(x, current_vt)
        return x + self.b
    
    def singular_sorting_loss(self, weights):
        difs = weights[:-1] - weights[1:] #The differences of each value and the one after
        difsum = tf.math.divide_no_nan(tf.math.reduce_sum(tf.nn.relu(difs)), #Sum of negative differences
                                       tf.math.reduce_sum(tf.cast(difs < 0, tf.float32))) #divided by the amount of negative differences (with 0 if dividing by 0)
        sumnegs = tf.math.divide_no_nan(tf.math.reduce_sum(tf.nn.relu(-weights)),  #Sum of negative weights
                                        tf.math.reduce_sum(tf.cast(weights < 0, tf.float32))) #divided by the amount of negative weights (with 0 if dividing by 0)
        return difsum + sumnegs

#Can be created using a finished SVDDense layer. Used to actually decrease the parameter count, and does not prune or compress anymore. 
class RecomposedDense(keras.layers.Layer):
    def __init__(self, units=32, weights = [], b = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.input_shape = 0
        self.w = weights
        self.b = b

    def build(self, input_shape):
        self.input_shape = input_shape

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(self.w), trainable=True, name="weights"
        )
        
        self.b = self.add_weight(
            shape=(self.units,), initializer=tf.constant_initializer(self.b), trainable=True, name="bias"
        )
        
    def call(self, inputs):

        #Temp test, try to immediately prune everything but 1
        # tf.cond(self.rank > 1, lambda: self.rank.assign_sub(1), lambda: self.rank)

        #Calculate loss for this layer
#        self.add_loss(0)
        x = tf.matmul(inputs, self.w)
        return x + self.b
    
class DecomposedDense(keras.layers.Layer):
    def __init__(self, units=32, u_sigma = [], vt = [], b = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.rank = 1 #Overwritten during build.
        self.input_shape = 0
        self.u_sigma = u_sigma
        self.vt = vt
        self.b = b

    def build(self, input_shape):
        self.input_shape = input_shape
        
        self.rank = int(tf.shape(self.u_sigma)[1])

        self.u_sigma = self.add_weight(
            shape=(input_shape[-1], self.rank), initializer=tf.constant_initializer(self.u_sigma), trainable=True, name="u"
        )

        if len(self.vt) == 1:
           self.vt = self.vt[0][0]

        self.vt = self.add_weight(
            shape=(self.rank, self.units), initializer=tf.constant_initializer(self.vt), trainable=True, name="vt"
            #shape=(int(self.rank), self.units), initializer='zeros', trainable=True, name="vt"
        )
        #self.vt.assign(vt)

        self.b = self.add_weight(
            shape=(self.units,), initializer=tf.constant_initializer(self.b), trainable=True, name="bias"
        )

        self.rank = tf.Variable(self.rank, trainable=False, dtype=tf.int32)

        
    def call(self, inputs):

        #Temp test, try to immediately prune everything but 1
        # tf.cond(self.rank > 1, lambda: self.rank.assign_sub(1), lambda: self.rank)

        #Calculate loss for this layer
#        self.add_loss(0)
        x = tf.matmul(inputs, self.u_sigma)
        x = tf.matmul(x, self.vt)
        return x + self.b