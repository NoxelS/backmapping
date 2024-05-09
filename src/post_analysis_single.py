import os
import pickle
import socket
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sympy import true

from library.classes.generators import FICDataGenerator, inverse_scale_output_ic
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN, MODEL_TYPES
from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, get_IC_max_index, ic_to_hlabel

##### CONFIGURATION #####

DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)
PADDING = int(config(Keys.PADDING))
CG_SIZE = (
    12 + 2 * PADDING,
    3 * (1 + NEIGHBORHOOD_SIZE) + 2 * PADDING,
    1,
)
OUTPUT_SIZE = (1 + 2 * PADDING, 1 + 2 * PADDING, 1)

# This is the maximum internal coordinate index
MAX_IC_INDEX = get_IC_max_index()

# Check if the internal coordinate index is valid
if len(sys.argv) < 2:
    raise Exception(f"Please provide the index of the internal coordinate that should be fitted as an argument, choose one out of: 0-{MAX_IC_INDEX}")

# Check if the internal coordinate index is valid
target_ic_index = int(sys.argv[1])
if target_ic_index < 0 or target_ic_index > MAX_IC_INDEX:
    raise Exception(f"Invalid ic index: {target_ic_index}, choose one of: 0-{MAX_IC_INDEX}")

# Check if the internal coordinate index is a free ic
target_ic = get_ic_from_index(target_ic_index)
if target_ic["fixed"]:
    raise ValueError(f"Internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)}) is not a free internal coordinate!")

SAVE_PATH = os.path.join(DATA_PREFIX, "analysis", "predictions_cache")

##### TRAINING #####

sample_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
)

train_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE,
)

validation_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE,
)


# The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
# performance when training on multiple GPUs.
strategy = tf.distribute.experimental.CentralStorageStrategy()

print(f"Starting to load and train the model for internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)})")

with strategy.scope():
    save_file_path = os.path.join(SAVE_PATH, f"predictions_{target_ic_index}.pkl")
    predictions = []

    # Load predictions if not done previously
    if not os.path.exists(save_file_path):

        # Check if path exists
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        cnn = CNN(
            CG_SIZE,
            OUTPUT_SIZE,
            data_prefix=DATA_PREFIX,
            display_name=f"{MODEL_NAME_PREFIX}_{target_ic_index}",
            keep_checkpoints=True,
            load_path=os.path.join(DATA_PREFIX, "models", str(target_ic_index), f"{MODEL_NAME_PREFIX}.h5"),
            # We currently use the keras MeanAbsoluteError loss function, because custom loss functions are not supproted while saving the model
            # in the current tensorflow version. This hopefully will change in the future.
            loss=BackmappingAbsolutePositionLoss(),
            test_sample=sample_gen.__getitem__(0),
            model_type=MODEL_TYPES.CNN_V2_0,
            ic_index=target_ic_index,
        )

        # Start calcualting the ic for all validation samples
        print("Starting to calculate the internal coordinates for the validation samples")

        predictions = []

        for i in range(len(train_gen)):
            x, y_true = train_gen.__getitem__(i)
            y_pred = cnn.model.predict(x)
            predictions.append((y_true, y_pred))

        # Save the predictions
        with open(save_file_path, "wb") as f:
            pickle.dump(predictions, f)

        # Load the predictions
        with open(save_file_path, "rb") as f:
            predictions = pickle.load(f)

    else:
        # Load the predictions
        with open(save_file_path, "rb") as f:
            predictions = pickle.load(f)

    true_ics = []
    pred_ics = []

    # Remove padding and loop over all samples in the batch and save in array
    for preds in predictions:
        y_true, y_pred = preds

        # Remove padding
        y_pred = y_pred[:, PADDING, PADDING, 0]
        y_true = y_true[:, PADDING, PADDING, 0]

        # Loop over all samples in the batch and save in array
        for j in range(y_pred.shape[0]):
            true_ics.append(y_true[j])
            pred_ics.append(y_pred[j])

    # Convert to numpy arrays
    true_ics = np.array(true_ics)
    pred_ics = np.array(pred_ics)

    # Reverse apply the scale function
    true_ics = inverse_scale_output_ic(target_ic_index, true_ics)
    pred_ics = inverse_scale_output_ic(target_ic_index, pred_ics)

    # Calculate mean and std
    true_mean = np.mean(true_ics)
    pred_mean = np.mean(pred_ics)
    true_std = np.std(true_ics)
    pred_std = np.std(pred_ics)

    # Print Mean
    print(f"True mean: {true_mean}")
    print(f"Pred mean: {pred_mean}")

    # Make histogram with the same bins
    bins = np.linspace(min(true_ics.min(), pred_ics.min()), max(true_ics.max(), pred_ics.max()), 150)

    # Plot the results as relative histogram
    plt.hist(true_ics, bins=bins, alpha=0.5, color="purple", label="True")
    plt.hist(pred_ics, bins=bins, alpha=0.5, color="green", label="Pred")

    # # Plot the mean and std
    # plt.axvline(true_mean, color="purple", linestyle="dashed", linewidth=1)
    # plt.axvline(pred_mean, color="green", linestyle="dashed", linewidth=1)

    # plt.axvline(true_mean + true_std, color="purple", linestyle="dotted", linewidth=1)
    # plt.axvline(true_mean - true_std, color="purple", linestyle="dotted", linewidth=1)

    # plt.axvline(pred_mean + pred_std, color="green", linestyle="dotted", linewidth=1)
    # plt.axvline(pred_mean - pred_std, color="green", linestyle="dotted", linewidth=1)

    # plt.legend(loc="upper right")
    plt.savefig(os.path.join(DATA_PREFIX, "analysis", f"ic_hist_{target_ic_index}.png"))
