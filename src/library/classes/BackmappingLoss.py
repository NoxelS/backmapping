import tensorflow as tf


class BackmappingLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))