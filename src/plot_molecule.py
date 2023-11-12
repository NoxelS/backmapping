import os
import sys
import socket

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from master import PORT, encode_finished, encode_starting
from library.config import Keys, config
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP
from library.static.vector_mappings import DOPC_AT_MAPPING
from library.classes.generators import PADDING_X, PADDING_Y, AbsolutePositionsNeigbourhoodGenerator, print_matrix, ABSOLUTE_POSITION_SCALE_Y

##### CONFIGURATION #####

LINE_MODE = False

DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

cg_size = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
at_size = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)
print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

# The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
# performance when training on multiple GPUs.
strategy = tf.distribute.experimental.CentralStorageStrategy()


# Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
atom_names_to_fit = [name for name in DOPC_AT_NAMES if not name.startswith("H")]

# Check which of those atoms already have a model
atom_names_to_fit_with_model = [name for name in atom_names_to_fit if os.path.exists(os.path.join(DATA_PREFIX, "models", f"{MODEL_NAME_PREFIX}{name}.h5"))]     # TODO: use CNN static function

print(f"{(atom_names_to_fit_with_model.__len__() / atom_names_to_fit.__len__() * 100 ):2f}% atoms without model")

##### PREDICTIONS #####

for sample_index in range(10):

    atom_position_predictions = []

    with strategy.scope():

        for atom_name in atom_names_to_fit_with_model:
            try:
                # Generator for test samples
                print(f"Starting loading data and training cache for atom {atom_name}")
                sample_gen = AbsolutePositionsNeigbourhoodGenerator(
                    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
                    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
                    input_size=cg_size,
                    output_size=at_size,
                    shuffle=False,
                    batch_size=1,
                    validate_split=VALIDATION_SPLIT,
                    validation_mode=True,               # <- We use the validation data here
                    augmentation=False,
                    only_fit_one_atom=True,
                    atom_name=atom_name,
                    neighbourhood_size=NEIGHBORHOOD_SIZE,
                    data_usage=DATA_USAGE
                )

                # Load model
                print(f"Loading model for atom {atom_name}")
                cnn = CNN(
                    cg_size,
                    at_size,
                    data_prefix=DATA_PREFIX,
                    display_name=atom_name,
                    keep_checkpoints=True,
                    load_path=os.path.join(DATA_PREFIX, "models", f"{MODEL_NAME_PREFIX}{atom_name}.h5"),
                    loss=BackmappingAbsolutePositionLoss(),
                    test_sample=sample_gen.__getitem__(sample_index),
                    socket=None,
                    host_ip_address=None,
                    port=None
                )

                # Test sample
                test_sample = sample_gen.__getitem__(sample_index)

                # Predict atom postion
                test_sample_pred = cnn.model.predict(test_sample[0])

                position = test_sample_pred[0, PADDING_X: - PADDING_X, PADDING_Y: -PADDING_Y, 0][0] * ABSOLUTE_POSITION_SCALE_Y        # In Angstrom
                true_position = test_sample[1][0, PADDING_X: - PADDING_X, PADDING_Y: -PADDING_Y, 0][0] * ABSOLUTE_POSITION_SCALE_Y     # In Angstrom
                data_point = {"atom_name": atom_name, "position": position, "loss": cnn.model.evaluate(test_sample[0], test_sample[1]), "input": test_sample[0][0], "output": true_position}

                atom_position_predictions.append(data_point)
                print(data_point["position"])
                print(data_point["position"].shape)

            except OSError:
                print(f"Could not load model for atom {atom_name}! Probably the model is currently being trained.")

        # Add N as origin to the data points
        atom_position_predictions.append({"atom_name": "N", "position": np.array([0, 0, 0]), "loss": [0, 0, 0], "input": np.zeros((1, cg_size[0], cg_size[1], cg_size[2])), "output": np.array([0, 0, 0])})


    print("Finished predicting atom positions!")

    # Print all predictions in the format atom_name: (x,y,z) of prediction
    for data_point in atom_position_predictions:
        print(f"{data_point['atom_name']}: {data_point['position']}")

    # Make a 3D scatter plot of the predictions
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Print all prediction points
    for data_point in atom_position_predictions:
        # Get hist file for the model
        hist_file = os.path.join(DATA_PREFIX, "hist", f"training_history_{data_point['atom_name']}.csv")
        total_epochs = 0

        # Get the total number of epochs
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                total_epochs = len(f.readlines())
        total_epochs = np.max([total_epochs - 2, 2])

        if total_epochs < 50:
            continue

        # Random color for atom
        color = DEFAULT_ELEMENT_COLOR_MAP[data_point["atom_name"][0]]

        if not LINE_MODE:
            # Print the predicted and true positions
            ax.scatter(data_point["position"][0], data_point["position"][1], data_point["position"][2], label=data_point["atom_name"], color=color)
            ax.scatter(data_point["output"][0], data_point["output"][1], data_point["output"][2], label=data_point["atom_name"]+"-true", color=color, alpha=0.25)

            # # Write the epoch number above the predicted position
            # ax.text(data_point["position"][0], data_point["position"][1], data_point["position"][2], f"$\mu={total_epochs}$", color='red', fontsize=6, alpha=0.5)
            
            # Draw lines between the predicted and true positions
            ax.plot([data_point["position"][0], data_point["output"][0]], [data_point["position"][1], data_point["output"][1]], [data_point["position"][2], data_point["output"][2]], color='black', linestyle='dashed', linewidth=1, alpha=0.1)

            # Write the distance between the predicted and true positions above the line
            ax.text((data_point["position"][0] + data_point["output"][0]) / 2, (data_point["position"][1] + data_point["output"][1]) / 2, (data_point["position"][2] + data_point["output"][2]) / 2, f"{np.linalg.norm(data_point['position'] - data_point['output']):.2f}Å", color='black', fontsize=6)

        # Print backbones connections of the molecule (predicted and true)
        for connection in DOPC_AT_MAPPING:
            from_atom_name = connection[0]
            to_atom_name = connection[1]

            if from_atom_name == data_point["atom_name"]:
                from_atom = data_point["position"]
                from_atom_name_predicted = data_point["output"]

                to_atoms = [atom for atom in atom_position_predictions if atom["atom_name"] == to_atom_name]

                if len(to_atoms) > 0:   # If there is no prediction for the atom, we can't draw a line
                    to_atom = to_atoms[0]["position"]
                    to_atom_predicted = to_atoms[0]["output"]

                    ax.plot([from_atom[0], to_atom[0]], [from_atom[1], to_atom[1]], [from_atom[2], to_atom[2]], color='blue', linestyle='solid', linewidth=1, alpha=0.25)
                    ax.plot([from_atom_name_predicted[0], to_atom_predicted[0]], [from_atom_name_predicted[1], to_atom_predicted[1]], [from_atom_name_predicted[2], to_atom_predicted[2]], color='blue', linestyle='dotted', linewidth=1, alpha=0.25)
            
            elif to_atom_name == data_point["atom_name"]:
                to_atom = data_point["position"]
                to_atom_name_predicted = data_point["output"]

                from_atoms = [atom for atom in atom_position_predictions if atom["atom_name"] == from_atom_name]

                if len(from_atoms) > 0:
                    from_atom = from_atoms[0]["position"]
                    from_atom_predicted = from_atoms[0]["output"]

                    ax.plot([from_atom[0], to_atom[0]], [from_atom[1], to_atom[1]], [from_atom[2], to_atom[2]], color='blue', linestyle='solid', linewidth=1, alpha=0.25)
                    ax.plot([from_atom_predicted[0], to_atom_name_predicted[0]], [from_atom_predicted[1], to_atom_name_predicted[1]], [from_atom_predicted[2], to_atom_name_predicted[2]], color='blue', linestyle='dotted', linewidth=1, alpha=0.25)

    # Add axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Add fig title
    ax.set_title(f"Predicted atomic positions for sample {sample_index}")

    # Save image
    plt.savefig(f"data/images/{sample_index}_predictions.png")