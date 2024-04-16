import os
import pickle
import time

import numpy as np

# Disable tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL

from library.classes.generators import ABSOLUTE_POSITION_SCALE, PADDING_X, PADDING_Y, NeighbourDataGenerator, get_scale_factor, print_matrix
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES, DOPC_CG_NAMES

##### CONFIGURATION #####

AUGMENT_DATA = False
VALIDATION_MODE = True

DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

ANALYSIS_PATH = os.path.join(DATA_PREFIX, "analysis")
ANALYSIS_PREDICTION_CACHE_PATH = os.path.join(ANALYSIS_PATH, "prediction_cache.pkl")
ANALYSIS_DATA_CACHE_PATH = os.path.join(ANALYSIS_PATH, "analysis_data_cache.pkl")

CG_SIZE = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
AT_SIZE = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)


def get_predictions(atom_names_to_fit_with_model, batch_size=2048, batches=1):
    """
    Generates predictions for a given set of atom names using a pre-trained CNN model.
    If the prediction cache is available and up-to-date, it will be loaded instead of re-generating the predictions.
    If the cache is outdated, it will be deleted and the predictions will be re-generated.

    Args:
        atom_names_to_fit_with_model (list): List of atom names to generate predictions for.
        sample_size (int, optional): Number of samples to generate predictions for. Defaults to 64.

    Returns:
        list: List of tuples containing the atom name, input data, true output data, predicted output data, and loss.
    """

    update_predictions = not os.path.exists(ANALYSIS_PREDICTION_CACHE_PATH)

    if not os.path.exists(ANALYSIS_PATH):
        os.makedirs(ANALYSIS_PATH, exist_ok=True)

    if os.path.exists(ANALYSIS_PREDICTION_CACHE_PATH):
        # Check if the prediction cache is older than the model folder OR generator cache files
        t_models = max(os.path.getmtime(root) for root, _, _ in os.walk(os.path.join(DATA_PREFIX, "models")))
        t_cache = os.path.getmtime(ANALYSIS_PREDICTION_CACHE_PATH)

        if t_models > t_cache:
            update_predictions = True
            os.remove(ANALYSIS_PREDICTION_CACHE_PATH)

    update_predictions = False

    if update_predictions:
        print("Starting running predictions, this might take a while...")

        # Delete old analysis data
        # TODO: handle this in the analysis data script
        if os.path.exists(ANALYSIS_DATA_CACHE_PATH):
            os.remove(ANALYSIS_DATA_CACHE_PATH)

        # List to store the predictions
        predictions = []

        import tensorflow as tf

        # The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
        # performance when training on multiple GPUs.
        STRATEGY = tf.distribute.experimental.CentralStorageStrategy()

        with STRATEGY.scope():

            for atom_name in atom_names_to_fit_with_model:
                try:
                    # Generator for test samples
                    print(f"Starting loading data and training cache for atom {atom_name}")
                    sample_gen = NeighbourDataGenerator(
                        input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
                        output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
                        input_size=CG_SIZE,
                        output_size=AT_SIZE,
                        shuffle=False,
                        batch_size=batch_size,
                        validate_split=VALIDATION_SPLIT,
                        validation_mode=VALIDATION_MODE,
                        augmentation=AUGMENT_DATA,
                        only_fit_one_atom=True,
                        atom_name=atom_name,
                        neighbourhood_size=NEIGHBORHOOD_SIZE,
                        data_usage=DATA_USAGE,
                    )

                    # Load model
                    print(f"Loading model for atom {atom_name}")
                    cnn = CNN(
                        CG_SIZE,
                        AT_SIZE,
                        data_prefix=DATA_PREFIX,
                        display_name=atom_name,
                        keep_checkpoints=True,
                        load_path=os.path.join(DATA_PREFIX, "models", atom_name, f"{MODEL_NAME_PREFIX}.h5"),
                        loss=tf.keras.losses.MeanAbsoluteError(),
                    )

                    # TODO: fix this with the batches so that we can use the whole dataset
                    for batch in range(batches):
                        # Load sample
                        test_sample = sample_gen.__getitem__(batch)  # This contains SAMPLE_N samples

                        X = test_sample[0]
                        Y_true = test_sample[1]
                        Y_pred = cnn.model.predict(X)
                        loss = cnn.model.evaluate(X, Y_true, verbose=0, return_dict=True)

                        # Add to list
                        predictions.append((atom_name, X, Y_true, Y_pred, loss))

                except OSError:
                    print(f"Could not load model for atom {atom_name}! Probably the model is currently being trained.")

        print("Finished running predictions, saving cache...")

        # Save predictions
        pickle.dump(predictions, open(ANALYSIS_PREDICTION_CACHE_PATH, "wb"))

        print(f"Saved prediction cache to {ANALYSIS_PREDICTION_CACHE_PATH}")

    # Load predictions
    predictions = pickle.load(open(ANALYSIS_PREDICTION_CACHE_PATH, "rb"))  # List of tuples (atom_name, X, Y_true, Y_pred, loss)
    print(f"Succesfully loaded prediction cache from {time.ctime(os.path.getmtime(ANALYSIS_PREDICTION_CACHE_PATH))}")

    return predictions


def predictions_to_analysis_data(predictions):
    """
        The predictions are a tuple of type (atom_name, X, Y_true, Y_pred, loss(dict) ) where Y_true and Y_pred contain the positions of ONE atom.
        We need to transform this into a list of molecules, where each molecule consists of a list of (X, Y_true, Y_pred) where Y and Y_true has the position of
        every atom in the molecule. This is necessary for the analysis.

    Args:
        predictions (list): List of type (atom_name, X, Y_true, Y_pred, loss(dict) ).

    Returns:
        list: List of molecules, where each molecule consists of a list of (X, Y_true, Y_pred) where Y and Y_true has the position of every atom in the molecule.
    """

    # TODO: also check if the cache is outdated
    if os.path.exists(ANALYSIS_DATA_CACHE_PATH):
        # Load analysis data
        analysis_data = pickle.load(open(ANALYSIS_DATA_CACHE_PATH, "rb"))
        print(f"Succesfully loaded analysis data cache from {time.ctime(os.path.getmtime(ANALYSIS_DATA_CACHE_PATH))}")
        return np.array(analysis_data)

    # Get the total number of molecules
    total_molecule = predictions[0][1].shape[0]

    # Check if every prediction has the same number of atom predictions
    for prediction in predictions:
        if prediction[1].shape[0] != total_molecule:
            raise ValueError("Not every prediction has the same number of atom predictions!")

    # List to store the analysis data
    analysis_data = []

    for i in range(total_molecule):
        # Get the X positions of the molecule
        X_poses = predictions[0][1][i, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]  # X is the same for every atom in the molecule

        # Scale up the X values to invert the scaling so that the absolute positions are again absolute in angstrom
        X_poses *= ABSOLUTE_POSITION_SCALE

        # Lists to store the positions of the atoms in the molecule
        X = []
        Y_pred = []
        Y_true = []

        # Add name of bead to X
        for j, x in enumerate(X_poses):
            X.append(
                (DOPC_CG_NAMES[j % 12], x.numpy())
            )  # j % 12 because there are 12 beads in a molecule and we need to repeat the names for every molecule and neighbor

        # Loop though every atom model prediction and add the one atom to the molecule
        for prediction in predictions:
            atom_name: str = prediction[0]
            Y_true_atom = prediction[2][i, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]
            Y_pred_atom = prediction[3][i, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]

            # Invert scaling
            Y_true_atom *= get_scale_factor(atom_name)
            Y_pred_atom *= get_scale_factor(atom_name)

            # Add to molecule
            Y_true.append((atom_name, *Y_true_atom.numpy()))
            Y_pred.append((atom_name, *Y_pred_atom))  # Y_pred is not a tensor, so we don't need to convert it to numpy

        analysis_data.append((np.array(X), np.array(Y_true), np.array(Y_pred)))

        # Print progress
        if i % 100 == 0:
            print(f"Processed molecule {i}/{total_molecule}")

    # Save analysis data
    pickle.dump(analysis_data, open(ANALYSIS_DATA_CACHE_PATH, "wb"))

    return np.array(analysis_data)
