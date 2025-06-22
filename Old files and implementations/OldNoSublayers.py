import tensorflow as tf
import numpy as np
from tensorflow import keras

#Very old, deprecated version. Left mostly for nostalgia.
class SVDDense(keras.layers.Layer):
    
    def __init__(self, units=32, lam_struct = 1, u_ort = 1e-10, u_sort = 1, lam_comp = 0.1, prune_threshold = 0.1  , layercount = 1, *args, **kwargs):
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
        vt = tf.transpose(v)
        self.rank = tf.Variable(int(tf.shape(sigma)[0]), trainable=False, dtype=tf.int32)

        
        self.u = self.add_weight(
            shape=(input_shape[-1], int(self.rank)), initializer=initializer, trainable=True, name="u"
        )
        self.u.assign(u)

        self.sigma = self.add_weight(
            shape=(int(self.rank),), initializer=initializer, trainable=True, name="sigma"
        )
        self.sigma.assign(sigma)

        self.vt = self.add_weight(
            shape=(int(self.rank), self.units), initializer=initializer, trainable=True, name="vt"
        )
        self.vt.assign(vt)

        self.b = self.add_weight(
            shape=(self.units,), initializer=initializer, trainable=True, name="bias"
        )


        # weights = self.get_weights()

        # newWeights = [np.diag(sig), u, vt, b]

        # for i in range(4):
        #     print("Layers:", i, "has shape", np.shape(weights[i]))
        #     print("np:", i, "has shape", np.shape(newEwights[i]))


    def call(self, inputs):

        #Temp test, try to immediately prune everything but 1
        # tf.cond(self.rank > 1, lambda: self.rank.assign_sub(1), lambda: self.rank)

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
                    tf.norm(tf.matmul(current_u, current_u, transpose_a=True) - tf.eye(tf.shape(current_sigma)[0])) + 
                    tf.norm(tf.matmul(current_vt, current_vt, transpose_b=True) - tf.eye(tf.shape(current_sigma)[0]))
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