import os
import pickle
import time

# Disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import tensorflow as tf

from library.classes.generators import (PADDING_X,
                                        PADDING_Y,
                                        AbsolutePositionsNeigbourhoodGenerator,
                                        get_scale_factor, print_matrix)
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from master import PORT, encode_finished, encode_starting

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

# The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
# performance when training on multiple GPUs.
STRATEGY = tf.distribute.experimental.CentralStorageStrategy()

CG_SIZE = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
AT_SIZE = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)


def get_analysis_data(atom_names_to_fit_with_model, sample_size = 64):
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
        t_models = max(os.path.getmtime(root) for root,_,_ in os.walk(os.path.join(DATA_PREFIX, "models")))
        t_cache = os.path.getmtime(ANALYSIS_PREDICTION_CACHE_PATH)
        
        if t_models > t_cache:
            update_predictions = True
            os.remove(ANALYSIS_PREDICTION_CACHE_PATH)

    if update_predictions:
        print("Starting running predictions, this might take a while...")

        # List to store the predictions
        predictions = []

        with STRATEGY.scope():

            for atom_name in atom_names_to_fit_with_model:
                try:
                    # Generator for test samples
                    print(f"Starting loading data and training cache for atom {atom_name}")
                    sample_gen = AbsolutePositionsNeigbourhoodGenerator(
                        input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
                        output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
                        input_size=CG_SIZE,
                        output_size=AT_SIZE,
                        shuffle=False,
                        batch_size=sample_size,
                        validate_split=VALIDATION_SPLIT,
                        validation_mode=VALIDATION_MODE,
                        augmentation=AUGMENT_DATA,
                        only_fit_one_atom=True,
                        atom_name=atom_name,
                        neighbourhood_size=NEIGHBORHOOD_SIZE,
                        data_usage=DATA_USAGE
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

                    # Load sample
                    test_sample = sample_gen.__getitem__(0)  # This contains SAMPLE_N samples

                    X = test_sample[0]
                    Y_true = test_sample[1]
                    Y_pred = cnn.model.predict(X)
                    loss   = cnn.model.evaluate(X, Y_true, verbose=0, return_dict=True)
                    
                    # Add to list
                    predictions.append((atom_name, X, Y_true, Y_pred, loss))

                except OSError:
                    print(f"Could not load model for atom {atom_name}! Probably the model is currently being trained.")

        print("Finished running predictions, saving cache...")

        # Save predictions
        pickle.dump(predictions, open(ANALYSIS_PREDICTION_CACHE_PATH, "wb"))
        
        print(f"Saved prediction cache to {ANALYSIS_PREDICTION_CACHE_PATH}")

    # Load predictions
    predictions = pickle.load(open(ANALYSIS_PREDICTION_CACHE_PATH, "rb"))   # List of tuples (atom_name, X, Y_true, Y_pred, loss)
    print(f"Succesfully loaded prediction cache from {time.ctime(os.path.getmtime(ANALYSIS_PREDICTION_CACHE_PATH))}")

    return predictions