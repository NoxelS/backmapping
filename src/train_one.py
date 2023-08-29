import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings

# from lib.cnn import CNN
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
from library.classes.models import CNN
from library.classes.generators import RelativeVectorsTrainingDataGenerator, AbsolutePositionsGenerator, PADDING_X, PADDING_Y, print_matrix
from library.parser import pdb_data_to_xyz, cg_xyz_to_pdb_data, at_xyz_to_pdb_data
from Bio.PDB import PDBParser
from library.classes.losses import BackmappingRelativeVectorLoss, BackmappingAbsolutePositionLoss
import matplotlib.pyplot as plt
from library.static.topologies import DOPC_AT_NAMES

##### CONFIGURATION #####

# data_prefix = "/data/users/noel/data/"        # For smaug
# data_prefix = "/localdisk/noel/"              # For fluffy
data_prefix = "data/"                           # For local

cg_size = (12 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
at_size = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)

BATCH_SIZE = 1024  # 16384 # 1024
VALIDATION_SPLIT = 0.1

print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

##### TRAINING #####

ATOM_NAME_TO_FIT = "C312"

sample_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    only_fit_one_atom=True,
    atom_name=ATOM_NAME_TO_FIT
)

cnn = CNN(
    cg_size,
    at_size,
    data_prefix=data_prefix,
    display_name="DOPC",
    keep_checkpoints=True,
    load_path="models/DOPC.h5",
    loss=BackmappingAbsolutePositionLoss(),
    test_sample=sample_gen.__getitem__(0),
)

train_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=True,
    only_fit_one_atom=True,
    atom_name=ATOM_NAME_TO_FIT
)

validation_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=True,
    only_fit_one_atom=True,
    atom_name=ATOM_NAME_TO_FIT
)

cnn.fit(
    train_gen,
    batch_size=BATCH_SIZE,
    epochs=200,
    validation_gen=validation_gen
)
