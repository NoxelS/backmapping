import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings

# from lib.cnn import CNN
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
from library.classes.models import CNN
from library.classes.generators import RelativeVectorsTrainingDataGenerator, PADDING_X, PADDING_Y
from library.parser import pdb_data_to_xyz, cg_xyz_to_pdb_data, at_xyz_to_pdb_data
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt

##### CONFIGURATION #####

# data_prefix = "/data/users/noel/data/"        # For smaug
# data_prefix = "/localdisk/noel/"              # For fluffy
data_prefix = "data/"                           # For local

cg_size = (11 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)
at_size = (53 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)

BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.1

print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

##### TRAINING #####

cnn = CNN(
    cg_size,
    at_size,
    data_prefix=data_prefix,
    display_name="DOPC",
    keep_checkpoints=True,
    load_path="models/DOPC.h5",
)

train_gen = RelativeVectorsTrainingDataGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False
)

validation_gen = RelativeVectorsTrainingDataGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True
)

cnn.fit(
    train_gen,
    batch_size=BATCH_SIZE,
    epochs=200,
    validation_gen=validation_gen,
)


test_gen = RelativeVectorsTrainingDataGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False
)

test_output = cnn.predict(test_gen.__getitem__(0)[0])
print(test_gen.__getitem__(0)[0][0, 10, :, 0])
print(test_output[0, 10, :, 0])

# Calculate mean squared error
mse = np.sqrt(np.mean(np.square(test_output * 100 - test_gen.__getitem__(0)[1] * 100)))
print(f"Mean squared error: {mse}")