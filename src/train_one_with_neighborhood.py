from library.classes.generators import RelativeVectorsTrainingDataGenerator, AbsolutePositionsGenerator, PADDING_X, PADDING_Y, print_matrix, AbsolutePositionsNeigbourhoodGenerator
from library.classes.losses import BackmappingRelativeVectorLoss, BackmappingAbsolutePositionLoss
from library.parser import pdb_data_to_xyz, cg_xyz_to_pdb_data, at_xyz_to_pdb_data
from library.static.topologies import DOPC_AT_NAMES
from scipy.ndimage import gaussian_filter
from library.classes.models import CNN
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import tensorflow as tf
import numpy as np
import sys
import os

##### CONFIGURATION #####

# data_prefix = "/data/users/noel/data/"        # For smaug
# data_prefix = "/localdisk/noel/"              # For fluffy
data_prefix = "data/"                           # For local

BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.1
NEIGHBORHOOD_SIZE = 6

cg_size = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
at_size = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)

print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

##### TRAINING #####

# Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
atom_names_to_fit = [name for name in DOPC_AT_NAMES if not name.startswith("H")]

if len(sys.argv) != 2:
    raise Exception(f"Please provide the name of the atom that should be fitted as an argument, choose one of: {', '.join(atom_names_to_fit)}")

atom_name_to_fit = sys.argv[1]

if atom_name_to_fit not in atom_names_to_fit:
    raise Exception(f"Invalid atom name: {atom_name_to_fit}, choose one of: {', '.join(atom_names_to_fit)}")


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
    load_path=os.path.join("models", atom_name_to_fit),
    loss=BackmappingAbsolutePositionLoss(),
    test_sample=sample_gen.__getitem__(0),
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
    epochs=1,
    validation_gen=validation_gen
)
