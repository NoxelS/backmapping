import tensorflow as tf
from library.classes.generators import BOX_SCALE_FACTOR, PADDING_X, PADDING_Y

class BackmappingLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Calculate the total bond direction loss
        bond_direction_loss = tf.reduce_mean(
            tf.square(y_true[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0] - y_pred[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]))
        
        # Calculate the total bond length loss
        bond_length_loss = []
        for vec_true, vec_pred in zip(y_true[:,  PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0], y_pred[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]):
            l_true = tf.norm(vec_true)
            l_pred = tf.norm(vec_pred)
            bond_length_loss.append(tf.square(l_true - l_pred))
            
        bond_length_loss = tf.math.sqrt(tf.reduce_mean(bond_length_loss))
        
        print(
            f" - bdl={bond_direction_loss:.4f} - bll={(bond_length_loss * BOX_SCALE_FACTOR):.4f}A")
            
        return bond_direction_loss + bond_length_loss
