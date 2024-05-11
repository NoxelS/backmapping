import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        ic_scaled_value_true = y_true[:, :, :, 0]
        ic_scaled_value_pred = y_pred[:, :, :, 0]

        # Add mse loss
        loss = tf.reduce_mean(tf.abs(ic_scaled_value_true - ic_scaled_value_pred) ** 2)

        # Calculate the total loss
        return loss
