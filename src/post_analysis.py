import logging
import os
import pickle
import socket
import sys
import time

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D

from library.analysis.data import get_predictions, predictions_to_analysis_data
from library.analysis.plots import (molecule_to_ptb,
                                    plot_bond_angle_distribution,
                                    plot_bond_length_distribution,
                                    plot_cluster_hist,
                                    plot_coordinates_distribution,
                                    plot_loss_atom_name, plot_loss_nmd,
                                    plot_molecule,
                                    plot_radial_distribution_function,
                                    plot_total_angle_distribution,
                                    plot_total_bond_length_distribution)
from library.classes.generators import (ABSOLUTE_POSITION_SCALE, PADDING_X,
                                        PADDING_Y, NeighbourDataGenerator,
                                        get_scale_factor, print_matrix)
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP, log
from library.static.vector_mappings import DOPC_AT_BAB, DOPC_AT_MAPPING
from master import PORT, encode_finished, encode_starting

##### CONFIGURATION #####

# Plot config
THEMES = ["seaborn-v0_8-paper", "seaborn-paper"]    # Use the first theme that is available

# Load config
DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

# Analysis config
N_BATCHES = 1
ANALYSIS_DATA_PATH = os.path.join(DATA_PREFIX, "analysis")

# Config of models
ATOM_NAMES_TO_FIT = [name for name in DOPC_AT_NAMES if not name.startswith("H")]  # Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
ATOM_NAMES_TO_FIT_WITH_MODEL = [name for name in ATOM_NAMES_TO_FIT if os.path.exists(os.path.join(DATA_PREFIX, "models", name, f"{MODEL_NAME_PREFIX}.h5"))]  # Check which of those atoms already have a model

# Matplotlib config
[plt.style.use(THEME) if THEME in plt.style.available else print(f"Theme '{THEME}' not available, using default theme. Select one of {plt.style.available}.") for THEME in THEMES]
savefig_kwargs = {"dpi": 300, "bbox_inches": 'tight'}

#### UTILS ####

def gen_path(*args):
    """
        Returns the path to the analysis data folder. Also creates the folder if it does not exist.
    """
    # Create folder if it does not exist
    if len(args) > 1:
        os.makedirs(os.path.join(ANALYSIS_DATA_PATH, *args[:-1]), exist_ok=True)

    return os.path.join(ANALYSIS_DATA_PATH, *args)


##### ANALYSIS #####

"""
    The predictions are a tuple of type (atom_name, X, Y_true, Y_pred, loss(dict) ).
    Note that the analysis data consists of the validation data and is generated if and only if the cache is not available or outdated.
"""
predictions = get_predictions(ATOM_NAMES_TO_FIT_WITH_MODEL, batch_size=BATCH_SIZE, batches=N_BATCHES)


"""
    Change the predictions into a more convenient format for analysis.
    The analysis data is a list of molecules, where each molecule consists of a list of (X, Y_true, Y_pred).
    X, Y_true and Y_pred are lists of tuples (atom_name, position).
"""
analysis_data = predictions_to_analysis_data(predictions)


"""
    The loss(atom_name) bar-chart plots give insights about the performance of the model for each atom.
"""
plot_loss_atom_name(predictions, 'loss').savefig(gen_path("loss", "loss_atom_name.png"), **savefig_kwargs)
plot_loss_atom_name(predictions, 'mae').savefig(gen_path("loss", "mae_atom_name.png"), **savefig_kwargs)


"""
    The loss(nmd) chart gives insights about the performance of the model with respect to the normalized
    mean distance of the atom a model fits.
"""
plot_loss_nmd(predictions).savefig(gen_path("loss", "loss_nmd.png"), **savefig_kwargs)


"""
    Plot the training history for each atom. This gives insights about the training process,
    cache behavior, and the training performance.
"""
plot_cluster_hist(2).savefig(gen_path("training", "training_loss.png"), **savefig_kwargs)
plot_cluster_hist(3).savefig(gen_path("training", "training_lr.png"), **savefig_kwargs)
plot_cluster_hist(4).savefig(gen_path("training", "training_mae.png"), **savefig_kwargs)
plot_cluster_hist(5).savefig(gen_path("training", "training_val_acc.png"), **savefig_kwargs)
plot_cluster_hist(6).savefig(gen_path("training", "training_val_loss.png"), **savefig_kwargs)
plot_cluster_hist(7).savefig(gen_path("training", "training_val_mae.png"), **savefig_kwargs)


"""
    This plot shows the predicted and true bond lengths as a histogram that overlaps for every bond in DOPC.
"""
[plot_bond_length_distribution(analysis_data, bond).savefig(gen_path("bonds", "lengths", f"bond_length_{bond[0]}_{bond[1]}.png"), **savefig_kwargs) for bond in DOPC_AT_MAPPING]


"""
    This plot shows the predicted and true bond lengths as a histogram.
"""
plot_total_bond_length_distribution(analysis_data).savefig(gen_path("bonds", "total_bond_length.png"), **savefig_kwargs)

"""
    This plot shows the predicted and true bond lengths as a histogram without the current misbehaving models
"""
plot_total_bond_length_distribution(analysis_data, skip_atoms=["C12", "C13", "C14"]).savefig(gen_path("bonds", "total_bond_length_ignoring_bad_models.png"), **savefig_kwargs)


"""
    Plot a few molecules as a 3D plot. This is useful to get a feeling for the data and the model performance.
    Also creates animation to visualize the 3D plot.
"""
[plot_molecule(analysis_data, i).savefig(gen_path(f"molecules", f"mol_{i}.png"), **savefig_kwargs) for i in range(25)]


"""
    Plot bond angel distribution of predicted and true bonds as a histogram.
"""
[plot_bond_angle_distribution(analysis_data, bond1, bond2).savefig(gen_path(f"bonds", "angles", f"bond_angle_{bond1[0]}_{bond1[1]}_{bond2[0]}_{bond2[1]}.png"), **savefig_kwargs) for bond1, bond2 in DOPC_AT_BAB]


"""
    Plot total bond angel distribution of predicted and true bonds as a histogram.
"""
plot_total_angle_distribution(analysis_data, DOPC_AT_BAB).savefig(gen_path(f"bonds", f"total_bond_angles.png"), **savefig_kwargs)



"""
    Plot bond dihedrial angle error of predicted and true bonds as a histogram.
"""
# TODO


"""
    Plot a coordinate distribution of predicted and true atom positions for every model.
    This is a chart with x,y,z coordinates on the x-axis and the frequency of atoms on the y-axis.
"""
[plot_coordinates_distribution(analysis_data, atom, 'x').savefig(gen_path("positions", "coordinates", f"coordinates_{atom}_x.png"), **savefig_kwargs) for atom in ATOM_NAMES_TO_FIT]
[plot_coordinates_distribution(analysis_data, atom, 'y').savefig(gen_path("positions", "coordinates", f"coordinates_{atom}_y.png"), **savefig_kwargs) for atom in ATOM_NAMES_TO_FIT]
[plot_coordinates_distribution(analysis_data, atom, 'z').savefig(gen_path("positions", "coordinates", f"coordinates_{atom}_z.png"), **savefig_kwargs) for atom in ATOM_NAMES_TO_FIT]



"""
    Transform a few predictions back into a PDB file to visualize and analyze the results with conventional tools.
"""
[molecule_to_ptb(analysis_data, 0, gen_path("PDB", f"mol_{i}_true.pdb"),gen_path("PDB", f"mol_{i}_pred.pdb"),gen_path("PDB", f"mol_{i}_CG.pdb")) for i in range(25)]


"""
    Plots a few neighbor predictions in one plot to visualize the performance of the model and also show
    the neighborhood of the atom.
"""
# TODO


"""
    Plot true and predicted radial distribution function g(r) for each model/atom
"""
[plot_radial_distribution_function(analysis_data, atom_name).savefig(gen_path("positions", "radial_distribution", f"g_{atom_name}.png"), **savefig_kwargs) for atom_name in ATOM_NAMES_TO_FIT]



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

