import numpy as np
import tensorflow as tf
from tensorflow import keras

#Custom madel as used in the field.
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cce_metric = keras.metrics.CategoricalCrossentropy(name="CCE")
        self.acc_metric = keras.metrics.Accuracy(name="acc")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            print(y)
            print(y_pred)
            prediction_loss = keras.losses.CategoricalCrossentropy().call(y, y_pred)
            
            if self.losses:
                loss = prediction_loss + tf.add_n(self.losses)
            else:
                loss = prediction_loss            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.cce_metric.update_state(y, y_pred)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "cce": self.cce_metric.result(), "acc": self.acc_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.cce_metric, self.acc_metric]
    
    def test_step(self, data):
            x, y = data
            y_pred = self(x, training=False)
            loss = keras.losses.CategoricalCrossentropy()(y, y_pred)

            # Update the metrics
            self.cce_metric.update_state(y, y_pred)
            self.acc_metric.update_state(y, y_pred)

            return {"cce": self.cce_metric.result(), "acc": self.acc_metric.result()}

#Version of the model that also includes two ranks in its return metrics. Should only be used on models with a minimum of two SVD layers, one of which has the name "first" and the other "second".
class CustomMetrics(keras.Model):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cce_metric = keras.metrics.CategoricalCrossentropy(name="CCE")
        self.acc_metric = keras.metrics.Accuracy(name="acc")
        self.rank_1 = keras.metrics.Mean(name="rank_1")
        self.rank_2 = keras.metrics.Mean(name="rank_2")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            print(y)
            print(y_pred)
            prediction_loss = keras.losses.CategoricalCrossentropy().call(y, y_pred)
            rank_1 = self.get_layer("first").rank
            self.rank_1.update_state(tf.reduce_mean(rank_1))
            rank_2 = self.get_layer("second").rank
            self.rank_2.update_state(tf.reduce_mean(rank_2))
            
            if self.losses:
                loss = prediction_loss + tf.add_n(self.losses)
            else:
                loss = prediction_loss            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.cce_metric.update_state(y, y_pred)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "cce": self.cce_metric.result(), "acc": self.acc_metric.result(), "rank_1": self.rank_1.result(), "rank_2": self.rank_2.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.cce_metric, self.acc_metric, self.rank_1, self.rank_2]

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = keras.losses.CategoricalCrossentropy()(y, y_pred)

        # Update the metrics
        self.cce_metric.update_state(y, y_pred)
        self.acc_metric.update_state(y, y_pred)

        return {"cce": self.cce_metric.result(), "acc": self.acc_metric.result()}