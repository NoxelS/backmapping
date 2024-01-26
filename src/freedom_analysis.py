import logging
import os
import pickle
import socket
import sys
import time

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from library.analysis.data import get_predictions, predictions_to_analysis_data
from library.analysis.utils import bond_length_distribution, bond_angle_distribution
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


def analyse_bond_lengths():

    THRESHOLD = 0.01

    bond_stds_mean_pairs = [bond_length_distribution(analysis_data, bond) for bond in DOPC_AT_MAPPING]

    bond_stds = [bond_stds_mean_pair[0] for bond_stds_mean_pair in bond_stds_mean_pairs]
    bond_means = [bond_stds_mean_pair[1] for bond_stds_mean_pair in bond_stds_mean_pairs]

    bond_relative_stds = [bond_std / bond_mean for bond_std, bond_mean in zip(bond_stds, bond_means)]

    # In percent
    bond_relative_stds = [bond_relative_std for bond_relative_std in bond_relative_stds]

    print(len(DOPC_AT_MAPPING))
    # Plot bond_std as scatter plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Bond Index")
    ax.set_ylabel("Relative Standard deviation")
    ax.set_title("Relative Bond Length Standard Deviation")
    ax.grid(True)

    # Print how many below threshold and how many above threshold
    below_threshold = np.count_nonzero(np.array(bond_relative_stds) < THRESHOLD)
    above_threshold = np.count_nonzero(np.array(bond_relative_stds) >= THRESHOLD)
    print(f"Below threshold: {below_threshold}")
    print(f"Above threshold: {above_threshold}")
    print(f"Percantage: {below_threshold / (below_threshold + above_threshold) * 100} %")


    # Plot line at y = THRESHOLD
    ax.axhline(THRESHOLD, color="red", linestyle="--", label="Threshold")

    # Write percentage of bonds below threshold on the threshold line
    ax.text(0, 1.05 * THRESHOLD, f"{(below_threshold / (below_threshold + above_threshold) * 100):02f} % bonds below threshold", color="black", fontsize=8)

    for i, bond in enumerate(DOPC_AT_MAPPING):
        ax.scatter(i, bond_relative_stds[i])

    ax.legend()
    fig.savefig(gen_path("bonds", "lengths", "bond_std.png"), **savefig_kwargs)


def analyse_bond_angles():

    THRESHOLD = 0.02

    bond_stds_mean_pairs = [bond_angle_distribution(analysis_data, bond1, bond2) for bond1, bond2 in DOPC_AT_BAB]

    bond_stds = [bond_stds_mean_pair[0] for bond_stds_mean_pair in bond_stds_mean_pairs]
    bond_means = [bond_stds_mean_pair[1] for bond_stds_mean_pair in bond_stds_mean_pairs]

    bond_relative_stds = [bond_std / bond_mean for bond_std, bond_mean in zip(bond_stds, bond_means)]

    # In percent
    bond_relative_stds = [bond_relative_std for bond_relative_std in bond_relative_stds]

    print(len(DOPC_AT_MAPPING))
    # Plot bond_std as scatter plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Bond Index")
    ax.set_ylabel("Relative Standard deviation")
    ax.set_title("Relative Bond Angle Standard Deviation")
    ax.grid(True)

    # Print how many below threshold and how many above threshold
    below_threshold = np.count_nonzero(np.array(bond_relative_stds) < THRESHOLD)
    above_threshold = np.count_nonzero(np.array(bond_relative_stds) >= THRESHOLD)
    print(f"Below threshold: {below_threshold}")
    print(f"Above threshold: {above_threshold}")
    print(f"Percantage: {below_threshold / (below_threshold + above_threshold) * 100} %")


    # Plot line at y = THRESHOLD
    ax.axhline(THRESHOLD, color="red", linestyle="--", label="Threshold")

    # Write percentage of bonds below threshold on the threshold line
    ax.text(0, 1.05 * THRESHOLD, f"{(below_threshold / (below_threshold + above_threshold) * 100):02f} % bonds below threshold", color="black", fontsize=8)

    for i, bond in enumerate(DOPC_AT_BAB):
        ax.scatter(bond_means[i], bond_relative_stds[i])

    ax.legend()
    fig.savefig(gen_path("bonds", "lengths", "angle_std.png"), **savefig_kwargs)


analyse_bond_lengths()
analyse_bond_angles()