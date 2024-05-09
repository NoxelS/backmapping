import tensorflow as tf

from library.classes.generators import PADDING


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # We ignore the padding
        ic_scaled_value_true = y_true[:, PADDING:-PADDING, PADDING:-PADDING, 0]
        ic_scaled_value_pred = y_pred[:, PADDING:-PADDING, PADDING:-PADDING, 0]

        # Add mse loss
        loss = tf.reduce_mean(tf.abs(ic_scaled_value_true - ic_scaled_value_pred) ** 2)

        # Calculate the total loss
        return loss
