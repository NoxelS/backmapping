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
at_size = (54 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors

BATCH_SIZE = 1024 # 16384 # 1024
VALIDATION_SPLIT = 0.1

print(f"Starting training with cg_size={cg_size} and at_size={at_size}")

##### TRAINING #####

sample_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=2,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False
)

cnn = CNN(
    cg_size,
    at_size,
    data_prefix=data_prefix,
    display_name="DOPC",
    keep_checkpoints=True,
    load_path="models/DOPC.h5",
    loss=BackmappingAbsolutePositionLoss(),
    test_sample=sample_gen.__getitem__(1035),
)

train_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=True
)

validation_gen = AbsolutePositionsGenerator(
    input_dir_path=os.path.join(data_prefix, "training", "input"),
    output_dir_path=os.path.join(data_prefix, "training", "output"),
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=True
)

cnn.fit(
    train_gen,
    batch_size=BATCH_SIZE,
    epochs=200,
    validation_gen=validation_gen
)

# test_index = 1

# test_output = cnn.predict(sample_gen.__getitem__(0)[0])[test_index:test_index+1, :, :, :]
# true_output = sample_gen.__getitem__(0)[1][test_index:test_index+1, :, :, :]

# # Plot the results in matplotlib
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(true_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0])
# axs[0].set_title("True")
# axs[1].imshow(test_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0])
# axs[1].set_title("Predicted")

# # Add colorbar
# fig.colorbar(axs[0].imshow(true_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]), ax=axs[0])
# fig.colorbar(axs[1].imshow(test_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]), ax=axs[1])

# # Write "x", "y", "z" to the x axis of the tree coordinates
# axs[0].set_xticks(np.arange(0, 3, 1))
# axs[0].set_xticklabels(["x", "y", "z"])
# axs[1].set_xticks(np.arange(0, 3, 1))
# axs[1].set_xticklabels(["x", "y", "z"])

# # Add labels to the y axis so every row has the name of the atom
# axs[0].set_yticks(np.arange(0, 54, 1))
# axs[1].set_yticks(np.arange(0, 54, 1))

# names_without_hydrogen = []
# for name in DOPC_AT_NAMES:
#     # If name starts with H, skip it
#     if name.startswith("H"):
#         continue
#     names_without_hydrogen.append(name)
    
# # Set y tick labels
# axs[0].set_yticklabels(names_without_hydrogen)
# axs[1].set_yticklabels(names_without_hydrogen)

# # Make the plot bigger
# fig.set_size_inches(10.5, 10.5)
# # plt.tight_layout(pad=0.05)

# # Retrive the positions from relative vectors
# positions_pred = []
# positions_true = []

# for i, bond in enumerate(test_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]):
#     positions_pred.append(bond)

# for i, bond in enumerate(true_output[0, PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y, 0]):
#     positions_true.append(bond)
    
# # Plot the positions in a 3D plot as points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter([x[0] for x in positions_true], [x[1] for x in positions_true], [x[2] for x in positions_true], c='b', marker='o')
# ax.scatter([x[0] for x in positions_pred], [x[1] for x in positions_pred], [x[2] for x in positions_pred], c='r', marker='o')
# # Add legend
# ax.legend(["True", "Predicted"])

# # Add labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Add title
# ax.set_title("Atom Positions")

# # Show the plot
# plt.show()


# # Calculate mean squared error
# mse = np.sqrt(np.mean(np.square(test_output * 100 - sample_gen.__getitem__(0)[1] * 100)))
# print(f"Mean squared error: {mse}")
