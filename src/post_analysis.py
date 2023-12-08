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
                                    plot_loss_nmd)
from library.classes.generators import (ABSOLUTE_POSITION_SCALE_X, PADDING_X,
                                        PADDING_Y,
                                        AbsolutePositionsNeigbourhoodGenerator,
                                        get_scale_factor, print_matrix)
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP, log
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

# Matplotlib config
plt.style.use(THEME) if THEME in plt.style.available else print(f"Theme '{THEME}' not available, using default theme. Select one of {plt.style.available}.")
savefig_kwargs = {
    "dpi": 300,
    "bbox_inches": 'tight'
}


##### ANALYSIS #####

"""
    The predictions are a tuple of type (atom_name, X, Y_true, Y_pred, loss(dict) ).
    Note that the analysis data consists of the validation data and is generated if and only if the cache is not available or outdated.
"""
predictions = get_analysis_data(ATOM_NAMES_TO_FIT_WITH_MODEL, batch_size=BATCH_SIZE, batches=N_BATCHES)


"""
    The loss(atom_name) bar-chart plots give insights about the performance of the model for each atom.
"""
log("Creating loss atom name plots...")
plot_loss_atom_name(predictions, 'loss').savefig("loss_atom_name.png", **savefig_kwargs)
plot_loss_atom_name(predictions, 'accuracy').savefig("acc_atom_name.png", **savefig_kwargs)
plot_loss_atom_name(predictions, 'mae').savefig("mae_atom_name.png", **savefig_kwargs)
log("Successfully created loss atom name plots.")

"""
    The loss(nmd) chart gives insights about the performance of the model with respect to the normalized
    mean distance of the atom a model fits.
"""
log("Creating loss nmd plot...")
plot_loss_nmd(predictions).savefig("loss_nmd.png", **savefig_kwargs)
log("Successfully created loss nmd plot.")

"""
    Plot the training history for each atom. This gives insights about the training process,
    cache behavior, and the training performance.
"""
log("Creating training history plots...")
plot_cluster_hist(2).savefig("training_loss.png", **savefig_kwargs)
plot_cluster_hist(3).savefig("training_lr.png", **savefig_kwargs)
plot_cluster_hist(4).savefig("training_mae.png", **savefig_kwargs)
plot_cluster_hist(5).savefig("training_val_acc.png", **savefig_kwargs)
plot_cluster_hist(6).savefig("training_val_loss.png", **savefig_kwargs)
plot_cluster_hist(7).savefig("training_val_mae.png", **savefig_kwargs)
log("Successfully created training history plots.")

"""
    This plot shows the predicted and true bond lengths as a histogram that overlaps
"""
log("Creating bond length distribution plot...")
plot_bond_length_distribution(predictions).savefig("bond_length_distribution.png", **savefig_kwargs)
log("Successfully created bond length distribution plot.")

"""
    Plot a few molecules as a 3D plot. This is useful to get a feeling for the data and the model performance.
    Also creates animation to visualize the 3D plot.
"""
# TODO


"""
    Plot bond angel distribution of predicted and true bonds as a histogram.
"""
# TODO


"""
    Plot bond dihedrial angle error of predicted and true bonds as a histogram.
"""
# TODO


"""
    Plot a coordinate distribution of predicted and true atom positions for every model.
    This is a chart with x,y,z coordinates on the x-axis and the frequency of atoms on the y-axis.
"""
# TODO


"""
    Transform a few predictions back into a PDB file to visualize and analyze the results with conventional tools.
"""
# TODO


"""
    Plots a few neighbor predictions in one plot to visualize the performance of the model and also show
    the neighborhood of the atom.
"""
# TODO


"""
    Plot true and predicted radial distribution function g(r) for each model/atom
"""
# TODO


"""
    Use the PDB file that was predicted from a fiven input and use gromacs to find out how long it takes
    until the energy is minimized. This is probably the best predictor of how good the model
    predicting power really ist.
"""
# TODO

"""
    Make a complete analysis of the model performance and create a report with all the plots and
    statistics, also generate a csv file with all the data that may be useful for further analysis.
"""
# TODO
