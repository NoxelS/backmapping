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

from library.classes.generators import (ABSOLUTE_POSITION_SCALE_X, PADDING_X,
                                        PADDING_Y,
                                        AbsolutePositionsNeigbourhoodGenerator,
                                        get_scale_factor, print_matrix, get_mean_distance_and_std)
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP
from library.static.vector_mappings import DOPC_AT_MAPPING
from library.analysis.data import get_analysis_data
from master import PORT, encode_finished, encode_starting

##### CONFIGURATION #####

# Analysis config
SAMPLE_SIZE = 64

# Plot config
THEME = "seaborn-v0_8-paper"
FIG_SIZE_RECT = (16, 9)
FIG_SIZE_SQUARE = (10, 10)

# Load config
DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

# Paths
PATH_TO_HIST = os.path.join(config(Keys.DATA_PATH), "hist")

# Config of models
ATOM_NAMES_TO_FIT = [name for name in DOPC_AT_NAMES if not name.startswith("H")]  # Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
ATOM_NAMES_TO_FIT_WITH_MODEL = [name for name in ATOM_NAMES_TO_FIT if os.path.exists(os.path.join(DATA_PREFIX, "models", name, f"{MODEL_NAME_PREFIX}.h5"))]  # Check which of those atoms already have a model

# Use python theme
plt.style.use(THEME) if THEME in plt.style.available else print(f"Theme '{THEME}' not available, using default theme. Select one of {plt.style.available}.")


### Plot functions ###

def plot_loss_atom_name(predictions, loss = "loss"):
    losses = [prediction[4][loss] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]
    
    # Make NMD colors
    nmd = np.array([get_mean_distance_and_std(prediction[0])[0] for prediction in predictions])
    nmd = nmd / np.max(nmd)
    colors = [plt.cm.cool(mean_distance) for mean_distance in nmd]

    # Sort everything by mean distance
    atom_names, losses, colors = zip(*sorted(zip(atom_names, losses, colors), key=lambda x: x[0]))

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)
    plt.bar(atom_names, losses, color=colors)
    title_name = loss.capitalize()
    plt.title(f"{title_name} for each model")
    plt.ylabel(title_name)
    plt.xlabel("Model name")
    plt.ylim(bottom=0)
    plt.xticks(rotation=90)
    
    return fig

def plot_loss_nmd(predictions):
    losses = [prediction[4]["loss"] for prediction in predictions]
    nmd = np.array([get_mean_distance_and_std(prediction[0])[0] for prediction in predictions])
    
    # Normalize
    nmd = nmd / np.max(nmd)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_SQUARE)
    plt.scatter(nmd, losses)
    plt.title(f"Loss for each model")
    plt.ylabel("Loss")
    plt.xlabel("Normalized mean distance of model")
    
    return fig


# Plot the training history of all models in a single plot
def plot_cluster_hist(data_col = 2):
    # Get mean distances to color the plot accordingly
    mean_distances = {}

    # Read csv
    with open(os.path.join(config(Keys.DATA_PATH), "mean_distances.csv"), "r") as f:
        # Skip header
        f.readline()

        for line in f.readlines():
            # Split line
            line = line.split(",")
            # Get mean distance
            mean_distances[line[0]] = float(line[1])

    # Normalize mean distances
    mean_distances = {k: v / max(mean_distances.values()) for k, v in mean_distances.items()}

    fig = plt.figure(figsize=FIG_SIZE_SQUARE)
    ax = fig.add_subplot(111)
    
    min_loss = 99999

    hist_files = os.listdir(PATH_TO_HIST)

    # Sort by average distance
    hist_files.sort(key=lambda x: mean_distances[x.split("_")[2].split(".")[0]])
    hist_files.reverse()

    # Data to find the average graph
    avg_data = []

    # Loop over all files in the hist folder
    for i, hist in enumerate(hist_files):
        # file is named: training_history_C1.csv
        atom_name = hist.split("_")[2].split(".")[0]
        mean_distance = mean_distances[atom_name]

        try:
            # Load csv
            hist = np.loadtxt(os.path.join(PATH_TO_HIST, hist), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        except Exception as e:
            continue

        # If only one row, add dimension
        if len(hist.shape) == 1:
            hist = hist.reshape(1, -1)

        # There are maybe multiple train cylces so reindex the epochs accordingly
        hist[:, 0] = np.arange(hist.shape[0])

        # Get scaling factor
        scale_factor = get_scale_factor(atom_name)

        if np.min(hist[:, data_col] * scale_factor) < min_loss:
            min_loss = np.min(hist[:, data_col] * scale_factor)

        # Plot
        color = plt.cm.cool(mean_distance)
        ax.plot(hist[:, 0] + 1, hist[:, data_col] * scale_factor, label=atom_name, color=color, alpha=(1 - 0.5 * mean_distance))

        # Add to average data
        avg_data.append(hist[:, data_col] * scale_factor)

    # Plot average avg_data = list for every atom with the history of the loss
    max_epoch = np.max([len(i) for i in avg_data])
    avg_data_y = [[] for _ in range(max_epoch)]
    for atom_data in avg_data:
        for epoch, data in enumerate(atom_data):
            avg_data_y[epoch].append(data)
    
    avg_data = np.array([np.mean(i) for i in avg_data_y])

    # Plot average
    ax.plot(np.arange(len(avg_data)) + 1, avg_data, label="Average", color="black", alpha=0.5)

    # Get name of data
    data_name = ["", "Accuracy", "MSE Loss (Å)", "Learning Rate", "Mean Average Error", "Val. Accuracy", "Val. MSE Loss (Å)", "Val. Mean Average Error"][data_col]

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(data_name)
    ax.set_title("Training History")

    # Make log scale
    ax.set_yscale("log")

    # Add 10 y-ticks between min and max
    ax.set_yticks(np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10))
    
    # Add 10 y-tick labels
    ax.set_yticklabels([f"{i:.2f}" for i in np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10)])
    
    # Add line where the minimum loss is
    ax.axhline(y=min_loss, color="black", linestyle="--", alpha=0.4)
    
    # Add label
    text_center_y = min_loss - 0.66 * (min_loss - ax.get_ylim()[0])
    text_center_x = ax.get_xlim()[1] / 2
    ax.text(text_center_x, text_center_y, f"Minimum Loss: {min_loss:.2f} Å", horizontalalignment='center', verticalalignment='bottom', fontsize=12, color="black", alpha=0.75)

    # Plot legend outside of plot in two columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    ax.get_legend().set_title("Atom Names")

    # Add legend that explains color to the bottom
    ax2 = fig.add_axes([0.93, 0.11, 0.2, 0.05])
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation='horizontal')
    ax2.set_title("NMD") # Normalized Mean Distance

    return fig



def plot_molecule(predictions, sample: int):
    """
    """
    pass

def plot_bond_length_distribution(predictions):
    """
    """
    pass

def plot_bond_angle_distribution(predictions):
    """
    """
    pass

def plot_dihedrial_angle_distribution(predictions):
    """
    """
    pass

def plot_coordinates_distribution(predictions, atom_name: str):
    """
    """
    pass

def plot_radial_distribution_function(predictions, atom_name: str):
    """
    """
    pass

def plot_N_molecules(predictions, N: int):
    """
    """
    pass