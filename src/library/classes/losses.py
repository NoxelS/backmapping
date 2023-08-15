import tensorflow as tf


class BackmappingLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Check if y_pred is made of nan values
        # if tf.math.is_nan(tf.reduce_mean(y_pred)):
        #     raise Exception("The network predicted nan values")

        return tf.reduce_mean(tf.square(y_true - y_pred))