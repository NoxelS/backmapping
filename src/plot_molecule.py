import os
import sys
import time
import socket
import logging


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ffmpeg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from master import PORT, encode_finished, encode_starting
from library.config import Keys, config
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP
from library.static.vector_mappings import DOPC_AT_MAPPING
from library.classes.generators import PADDING_X, PADDING_Y, NeighbourDataGenerator, print_matrix, get_scale_factor, ABSOLUTE_POSITION_SCALE

##### CONFIGURATION #####

# Plot configuration
LINE_MODE = False
SAMPLE_SIZE = 64
AUGMENT_DATA = False
VALIDATION_MODE = True
MAKE_VIDEO = True

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

# The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
# performance when training on multiple GPUs.
strategy = tf.distribute.experimental.CentralStorageStrategy()

# Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
atom_names_to_fit = [name for name in DOPC_AT_NAMES if not name.startswith("H")]

# Check which of those atoms already have a model
atom_names_to_fit_with_model = [name for name in atom_names_to_fit if os.path.exists(os.path.join(DATA_PREFIX, "models", name, f"{MODEL_NAME_PREFIX}.h5"))]     # TODO: use CNN static function

##### PREDICTIONS #####
def del_dir(rootdir):
    for (dirpath, dirnames, filenames) in os.walk(rootdir):
        for filename in filenames: os.remove(rootdir+'/'+filename)
        for dirname in dirnames: del_dir(rootdir+'/'+dirname)
    os.rmdir(rootdir)

image_data = [[] for _ in range(SAMPLE_SIZE)]   # For every sample we store the data in a list
print(f"Generating predictions for {len(image_data)} samples")

time_start = time.time()
progress_i = 0

with strategy.scope():

    for atom_name in atom_names_to_fit_with_model:
        try:
            # Generator for test samples
            print(f"Starting loading data and training cache for atom {atom_name}")
            sample_gen = NeighbourDataGenerator(
                input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
                output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
                input_size=cg_size,
                output_size=at_size,
                shuffle=False,
                batch_size=SAMPLE_SIZE,
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
                cg_size,
                at_size,
                data_prefix=DATA_PREFIX,
                display_name=atom_name,
                keep_checkpoints=True,
                load_path=os.path.join(DATA_PREFIX, "models", atom_name, f"{MODEL_NAME_PREFIX}.h5"),
                loss=tf.keras.losses.MeanAbsoluteError(),
            )

            # Test sample
            test_sample = sample_gen.__getitem__(0)  # This contains SAMPLE_N samples

            # Predict atom postion
            test_sample_pred = cnn.model.predict(test_sample[0])
            test_sample_true = test_sample[1]

            # Get the scale factor for the atom
            scale_factor = get_scale_factor(atom_name)
        
            # Calculate the position of the atom
            positions = test_sample_pred[:, PADDING_X: - PADDING_X, PADDING_Y: -PADDING_Y, 0] * scale_factor        # In Angstrom
            true_position = test_sample_true[:, PADDING_X: - PADDING_X, PADDING_Y: -PADDING_Y, 0] * scale_factor     # In Angstrom
            
            # Remove the x dimension as it is always 0
            positions = positions[:, 0, :]
            true_position = true_position[:, 0, :]
            
            # Get the backbone atom positions of the neighbor molecules (from the coarse grain input)
            neighbor_positions = test_sample[0][:, PADDING_X + 12 : - PADDING_X, PADDING_Y: -PADDING_Y, 0] * ABSOLUTE_POSITION_SCALE
            
            
            for sample in range(SAMPLE_SIZE):
                image_data[sample].append({
                    "atom_name": atom_name, 
                    "position": positions[sample], 
                    "loss": cnn.model.evaluate(test_sample[0], test_sample[1]), # List of type [loss, mae, mse]
                    "input": test_sample[0][sample], 
                    "output": true_position[sample],
                    "neighbor_positions": neighbor_positions[sample]
                })
                
            # Print progress
            print(f"{progress_i}/{len(atom_names_to_fit_with_model)} Finished loading data and training cache for atom {atom_name}")
            progress_i += 1

        except OSError:
            print(f"Could not load model for atom {atom_name}! Probably the model is currently being trained.")

print(f"Finished predicting atom positions in {time.time() - time_start:.2f}s for {SAMPLE_SIZE} samples")

# Add N as origin to every sample
for sample in range(SAMPLE_SIZE):
    image_data[sample].append({"atom_name": "N", "position": np.array([0, 0, 0]), "loss": [0, 0, 0], "input": np.zeros((1, cg_size[0], cg_size[1], cg_size[2])), "output": np.array([0, 0, 0])})

####### PLOTTING ########

for data_i, data in enumerate(image_data):

    # Make a 3D scatter plot of the predictions
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Print all prediction points
    for data_point in data:
        # Get hist file for the model
        hist_file = os.path.join(DATA_PREFIX, "hist", f"training_history_{data_point['atom_name']}.csv")
        total_epochs = 0

        # Get the total number of epochs
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                total_epochs = len(f.readlines())
        total_epochs = np.max([total_epochs - 2, 2])

        # Get colors for atoms
        color = DEFAULT_ELEMENT_COLOR_MAP[data_point["atom_name"][0]]

        if not LINE_MODE:
            # Print the predicted and true positions
            ax.scatter(data_point["position"][0], data_point["position"][1], data_point["position"][2], label=data_point["atom_name"], color=color, alpha=0.8)
            ax.scatter(data_point["output"][0], data_point["output"][1], data_point["output"][2], label=data_point["atom_name"]+"-true", color=color, alpha=0.25)

            # # Write the epoch number above the predicted position
            # ax.text(data_point["position"][0], data_point["position"][1], data_point["position"][2], f"$\mu={total_epochs}$", color='red', fontsize=6, alpha=0.5)
            
            # Draw lines between the predicted and true positions
            ax.plot([data_point["position"][0], data_point["output"][0]], [data_point["position"][1], data_point["output"][1]], [data_point["position"][2], data_point["output"][2]], color='black', linewidth=0.1, alpha=0.1)

            # # Write the distance between the predicted and true positions above the line
            # ax.text((data_point["position"][0] + data_point["output"][0]) / 2, (data_point["position"][1] + data_point["output"][1]) / 2, (data_point["position"][2] + data_point["output"][2]) / 2, f"{np.linalg.norm(data_point['position'] - data_point['output']):.2f}Å", color='black', fontsize=6)

    # Plot neighbor molecules
    for neighbor in data[0]["neighbor_positions"]:
        # Plot 3D spheres for each neighbor molecule
        ax.scatter(neighbor[0], neighbor[1], neighbor[2], color='orange', alpha=0.3, s=100)
        
    # Draw line between beads in neihbor molecules (one molecule has 12 beds)
    # TODO
    
    # Print backbones connections of the molecule (predicted and true)
    for connection in DOPC_AT_MAPPING:
        from_atom = connection[0]
        to_atom = connection[1]
        
        # Get the positions of the atoms
        from_atom_position_pred = [i["position"] for i in data if i["atom_name"] == from_atom][0]
        to_atom_position_pred = [i["position"] for i in data if i["atom_name"] == to_atom][0]
        
        from_atom_position_true = [i["output"] for i in data if i["atom_name"] == from_atom][0]
        to_atom_position_true = [i["output"] for i in data if i["atom_name"] == to_atom][0]
        
        # Draw the line
        ax.plot([from_atom_position_pred[0], to_atom_position_pred[0]], [from_atom_position_pred[1], to_atom_position_pred[1]], [from_atom_position_pred[2], to_atom_position_pred[2]], color="blue", linewidth=0.5, linestyle='--', alpha=0.5)
        ax.plot([from_atom_position_true[0], to_atom_position_true[0]], [from_atom_position_true[1], to_atom_position_true[1]], [from_atom_position_true[2], to_atom_position_true[2]], color="purple", linewidth=0.5, linestyle='--', alpha=0.3)
    
        
    # # Write text-box for all relevant model information
    # text_box = f"Model: {MODEL_NAME_PREFIX}\n"
    # text_box += f"Epochs: {EPOCHS}\n"
    # text_box += f"Batch size: {BATCH_SIZE}\n"
    # text_box += f"Neighborhood size: {NEIGHBORHOOD_SIZE}\n"
    # text_box += f"Data usage: {DATA_USAGE}\n"
    # text_box += f"Validation split: {VALIDATION_SPLIT}\n"
    # text_box += f"Total epochs: {total_epochs}\n"
    
    # # Add text box
    # ax.text2D(0.05, 0.95, text_box, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Only plot from max to min of position data, the neighbor molecules are only plotted if they are in this range
    max_x = np.max([np.max([i["position"][0] for i in data]), np.max([i["output"][0] for i in data])])
    min_x = np.min([np.min([i["position"][0] for i in data]), np.min([i["output"][0] for i in data])])
    max_y = np.max([np.max([i["position"][1] for i in data]), np.max([i["output"][1] for i in data])])
    min_y = np.min([np.min([i["position"][1] for i in data]), np.min([i["output"][1] for i in data])])
    max_z = np.max([np.max([i["position"][2] for i in data]), np.max([i["output"][2] for i in data])])
    min_z = np.min([np.min([i["position"][2] for i in data]), np.min([i["output"][2] for i in data])])
    
    # Set axes limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    # Add axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    # Custom legend
    lines = [
        Line2D([0], [0], alpha=0.7, color='blue', linewidth=3, linestyle='--'),
        Line2D([0], [0], alpha=0.3, color='purple', linewidth=3, linestyle='--'),
        Line2D([0], [0], alpha=0.3, color='orange', linewidth=3, linestyle='solid')
    ]
    labels = ['predicted', 'groundtruth', 'neighbor']
    plt.legend(lines, labels)

    # Add fig title
    ax.set_title(f"Predicted {data_i}")

    # Save image
    plt.savefig(f"data/images/{data_i}_predictions.png")
    
    progress_i = 0
    if MAKE_VIDEO:
        # Make folder for animation
        os.makedirs(f"data/images/{data_i}_animation", exist_ok=True)
        
        # Make animation
        index = 0
        for i in range(0, 360, 1):
            ax.view_init(30, i)
            plt.savefig(f"data/images/{data_i}_animation/{index}.png")
            index += 1
            
        # Compile animation
        (
            ffmpeg
            .input(f"data/images/{data_i}_animation/%d.png", framerate=15)
            .output(f"data/images/{data_i}_animation.mp4")
            # .run()
            .run_async()
        )
        
        # Print progress
        print(f"{progress_i}/{len(image_data)} Finished plotting predictions for sample {data_i}")
        progress_i += 1

    # Clear plot
    plt.clf()