import tensorflow as tf

from library.classes.generators import PADDING, PADDING


class BackmappingAbsolutePositionLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_abs_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # We ignore the padding
        pos_true = y_true[:, PADDING:-PADDING, PADDING:-PADDING, 0]
        pos_pred = y_pred[:, PADDING:-PADDING, PADDING:-PADDING, 0]
        
        # Add loss for each atom position
        positional_loss = tf.reduce_mean(tf.norm(pos_true - pos_pred))

        # Calculate the total position loss
        return positional_loss
