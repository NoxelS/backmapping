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


##### SOCKET #####

use_socket = len(sys.argv) > 2

if use_socket:
    host_ip_address = sys.argv[2]
    print(f"Trying to connect to parent {host_ip_address}")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host_ip_address, PORT))
    client.send(encode_starting(atom_name_to_fit))
    client.close()
else:
    print("No host ip address provided, not using socket")


##### TRAINING #####

sample_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    only_fit_one_atom=True,
    atom_name=atom_name_to_fit,
    neighbourhood_size=NEIGHBORHOOD_SIZE
)

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

validation_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=False,
    only_fit_one_atom=True,
    atom_name=atom_name_to_fit,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE
)

with strategy.scope():

    cnn = CNN(
        cg_size,
        at_size,
        data_prefix=DATA_PREFIX,
        display_name=atom_name_to_fit,
        keep_checkpoints=True,
        load_path=os.path.join("data", "models", f"{MODEL_NAME_PREFIX}{atom_name_to_fit}.h5"),
        loss=BackmappingAbsolutePositionLoss(),
        test_sample=sample_gen.__getitem__(0),
        socket=client if use_socket else None,
        host_ip_address=host_ip_address if use_socket else None,
        port=PORT if use_socket else None
    )

    cnn.fit(
        train_gen,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_gen=validation_gen,
        use_tensorboard=USE_TENSORBOARD
    )

    cnn.save()

# Send finished signal
if use_socket:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host_ip_address, PORT))
    client.send(encode_finished(atom_name_to_fit))
    client.close()