import logging
import os
import pickle
import socket
import sys
import time

# Disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D

from library.analysis.data import get_analysis_data
from library.analysis.plots import (plot_bond_length_distribution,
                                    plot_cluster_hist, plot_loss_atom_name,
                                    plot_loss_nmd,
                                    plot_bond_angle_distribution,
                                    plot_bond_dihedral_angle_distribution)
from library.classes.generators import (ABSOLUTE_POSITION_SCALE_X, PADDING_X,
                                        PADDING_Y,
                                        AbsolutePositionsNeigbourhoodGenerator,
                                        get_scale_factor, print_matrix)
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP
from library.static.vector_mappings import DOPC_AT_MAPPING
from master import PORT, encode_finished, encode_starting

##### CONFIGURATION #####

# Analysis config
N_BATCHES = 15

# Plot config
THEME = "seaborn-v0_8-paper"

# Load config
DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

# Config of models
ATOM_NAMES_TO_FIT = [name for name in DOPC_AT_NAMES if not name.startswith("H")]  # Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
ATOM_NAMES_TO_FIT_WITH_MODEL = [name for name in ATOM_NAMES_TO_FIT if os.path.exists(os.path.join(DATA_PREFIX, "models", name, f"{MODEL_NAME_PREFIX}.h5"))]  # Check which of those atoms already have a model

# Use python theme
plt.style.use(THEME) if THEME in plt.style.available else print(f"Theme '{THEME}' not available, using default theme. Select one of {plt.style.available}.")



##### ANALYSIS #####

# Load predictions from cache or generate new ones
predictions = get_analysis_data(ATOM_NAMES_TO_FIT_WITH_MODEL, batch_size=BATCH_SIZE, batches=N_BATCHES)
print(len(predictions))
   
# Make loss(atom_bame) bar-chart plot
plot_loss_atom_name(predictions, 'loss').savefig("loss_atom_name.png", dpi=300, bbox_inches='tight')
plot_loss_atom_name(predictions, 'accuracy').savefig("acc_atom_name.png", dpi=300, bbox_inches='tight')
plot_loss_atom_name(predictions, 'mae').savefig("mae_atom_name.png", dpi=300, bbox_inches='tight')

# Make loss(nmd) scatter plot
plot_loss_nmd(predictions).savefig("loss_nmd.png", dpi=300, bbox_inches='tight')

# Plot training evolution for different metrics
plot_cluster_hist(2).savefig("training_loss.png", dpi=300, bbox_inches='tight')
plot_cluster_hist(3).savefig("training_lr.png", dpi=300, bbox_inches='tight')
plot_cluster_hist(4).savefig("training_mae.png", dpi=300, bbox_inches='tight')
plot_cluster_hist(5).savefig("training_val_acc.png", dpi=300,  bbox_inches='tight')
plot_cluster_hist(6).savefig("training_val_loss.png", dpi=300, bbox_inches='tight')
plot_cluster_hist(7).savefig("training_val_mae.png", dpi=300, bbox_inches='tight')

# Plot distributions
plot_bond_length_distribution(predictions).savefig("bond_length_distribution.png", dpi=300, bbox_inches='tight')
plot_bond_angle_distribution(predictions).savefig("bond_length_distribution.png", dpi=300, bbox_inches='tight')
plot_bond_dihedral_angle_distribution(predictions).savefig("bond_length_distribution.png", dpi=300, bbox_inches='tight')