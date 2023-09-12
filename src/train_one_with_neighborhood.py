import os
import sys
import socket

from master import PORT, encode_finished, encode_starting
from library.config import Keys, config
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.static.topologies import DOPC_AT_NAMES
from library.classes.generators import PADDING_X, PADDING_Y, AbsolutePositionsNeigbourhoodGenerator

##### CONFIGURATION #####

data_prefix = config(Keys.DATA_PATH)

BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)

cg_size = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
at_size = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)

print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

# Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
atom_names_to_fit = [name for name in DOPC_AT_NAMES if not name.startswith("H")]

if len(sys.argv) != 3:
    raise Exception(f"Please provide the name of the atom that should be fitted as an argument, choose one of: {', '.join(atom_names_to_fit)}")

atom_name_to_fit = sys.argv[1]
host_ip_address = sys.argv[2]

if atom_name_to_fit not in atom_names_to_fit:
    raise Exception(f"Invalid atom name: {atom_name_to_fit}, choose one of: {', '.join(atom_names_to_fit)}")

##### SOCKET #####

print(f"Trying to connect to parent {host_ip_address}")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host_ip_address, PORT))
client.send(encode_starting(atom_name_to_fit))
client.close()

##### TRAINING #####

sample_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
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

cnn = CNN(
    cg_size,
    at_size,
    data_prefix=data_prefix,
    display_name=atom_name_to_fit,
    keep_checkpoints=True,
    load_path=os.path.join("data", "models", f"{atom_name_to_fit}.h5"),
    loss=BackmappingAbsolutePositionLoss(),
    test_sample=sample_gen.__getitem__(0),
    socket=client,
    host_ip_address=host_ip_address,
    port=PORT,
)

train_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=True,
    only_fit_one_atom=True,
    atom_name=atom_name_to_fit,
    neighbourhood_size=NEIGHBORHOOD_SIZE
)

validation_gen = AbsolutePositionsNeigbourhoodGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=True,
    only_fit_one_atom=True,
    atom_name=atom_name_to_fit,
    neighbourhood_size=NEIGHBORHOOD_SIZE
)

cnn.fit(
    train_gen,
    batch_size=BATCH_SIZE,
    epochs=25,
    validation_gen=validation_gen
)

cnn.save()

# Send finished signal
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host_ip_address, PORT))
client.send(encode_finished(atom_name_to_fit))
client.close()