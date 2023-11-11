import os
import sys
import socket

import tensorflow as tf

from master import PORT, encode_finished, encode_starting
from library.config import Keys, config
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.static.topologies import DOPC_AT_NAMES
from library.classes.generators import PADDING_X, PADDING_Y, AbsolutePositionsNeigbourhoodGenerator

##### CONFIGURATION #####

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

if len(sys.argv) < 2:
    raise Exception(f"Please provide the name of the atom that should be fitted as an argument, choose one of: {', '.join(atom_names_to_fit)}")

atom_name_to_fit = sys.argv[1]

if atom_name_to_fit not in atom_names_to_fit:
    raise Exception(f"Invalid atom name: {atom_name_to_fit}, choose one of: {', '.join(atom_names_to_fit)}")

##### ANALYSING #####

train_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=True,
    only_fit_one_atom=True,
    atom_name=atom_name_to_fit,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE
)

positions = []

for i in range(len(train_gen)):
    x, y = train_gen[i]
    print(f"Batch {i}/{len(train_gen)}: {x.shape} -> {y.shape}")
    
    # Find every coordinate in the output for each batch and analyze the max, min and mean of coordinates
    for data_set in y:
        atom_pos = data_set[PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0][0]
        positions.append(abs(atom_pos[0].numpy()))
        positions.append(abs(atom_pos[1].numpy()))
        positions.append(abs(atom_pos[2].numpy()))

    print(f"\t Max: {max(positions)}")
    print(f"\t Min: {min(positions)}")
    print(f"\t Mean: {sum(positions) / len(positions)}")

print(f"-> Max: {max(positions)}")
print(f"-> Min: {min(positions)}")
print(f"-> Mean: {sum(positions) / len(positions)}")

