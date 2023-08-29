import tensorflow as tf
from library.classes.generators import BOX_SCALE_FACTOR, PADDING_X, PADDING_Y, print_matrix, ABSOLUT_POSITION_SCALE
class BackmappingRelativeVectorLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_rel_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Remove padding as we don't want to calculate the loss for the padding
        y_true = y_true[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]
        y_pred = y_pred[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]

        # Calculate the total bond direction loss
        y_true_n = tf.nn.l2_normalize(y_true, axis=-1) # Normalize the vectors to unit length
        y_pred_n = tf.nn.l2_normalize(y_pred, axis=-1) # Normalize the vectors to unit length
        bond_direction_loss = tf.reduce_mean(tf.square(y_true_n - y_pred_n))

        # Calculate the total bond length loss
        bond_length_loss = []
        for vec_true, vec_pred in zip(y_true, y_pred):
            l_true = tf.norm(vec_true) # Calculate the length in anstrom so we have a feeling for the loss
            l_pred = tf.norm(vec_pred) # Calculate the length in anstrom so we have a feeling for the loss
            bond_length_loss.append(tf.abs(l_true - l_pred))
        bond_length_loss = tf.reduce_mean(bond_length_loss)

        # Calculate the atom positions
        positions_pred = [tf.constant([0,0,0], dtype=tf.float32)]
        positions_true = [tf.constant([0,0,0], dtype=tf.float32)]
        for i, bond in enumerate(y_pred):
            positions_pred.append(tf.add(positions_pred[i], bond))
        for i, bond in enumerate(y_true):
            positions_true.append(tf.add(positions_true[i], bond))

        # Calculate the total position loss
        position_loss = []
        for pos_true, pos_pred in zip(positions_true, positions_pred):
            position_loss.append(tf.norm(pos_true - pos_pred))
        position_loss = tf.reduce_mean(position_loss)

        # # Print last two atoms
        # print(f"\n -> True: {positions_true[-1][-1] * BOX_SCALE_FACTOR} Pred: {positions_pred[-1][-1] * BOX_SCALE_FACTOR}")
        # # Print the losses
        # print(f" -> bdl={bond_direction_loss:.4f} - bll={(bond_length_loss * BOX_SCALE_FACTOR):.4f}A - pl={(position_loss * BOX_SCALE_FACTOR):.4f}A")

        return 0.3 * bond_direction_loss + 0.1 * bond_length_loss + 0.6 * position_loss


class BackmappingAbsolutePositionLoss(tf.keras.losses.Loss):
    def __init__(self, name="backmapping_abs_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Print first two datasets
        # print("\n")
        # print("True:")
        # print_matrix(y_true[0:1, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y,:])
        # print("Pred:")
        # print_matrix(y_pred[0:1, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, :])
        # print("Diff:")
        # print_matrix(y_true[0:1, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, :] - y_pred[0:1, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, :])
        
        # We ignore the padding
        pos_true = y_true[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]
        pos_pred = y_pred[:, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]
        
        # Add loss for each atom position
        positional_loss = tf.reduce_mean(tf.norm(pos_true - pos_pred))
        
        print(f" - pl: {(positional_loss * ABSOLUT_POSITION_SCALE):.4f}A, true: {pos_true[0] * ABSOLUT_POSITION_SCALE}, pred: {pos_pred[0] * ABSOLUT_POSITION_SCALE}")

        # Calculate the total position loss
        return positional_loss
