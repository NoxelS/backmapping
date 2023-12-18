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

from library.analysis.data import get_predictions
from library.classes.generators import (ABSOLUTE_POSITION_SCALE, PADDING_X,
                                        PADDING_Y, NeighbourDataGenerator,
                                        get_mean_distance_and_std,
                                        get_scale_factor, print_matrix)
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.config import Keys, config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP, log_progress
from library.static.vector_mappings import DOPC_AT_MAPPING
from master import PORT, encode_finished, encode_starting

##### CONFIGURATION #####

# Analysis config
SAMPLE_SIZE = 64

# Plot config
THEME = "seaborn-paper" #"seaborn-v0_8-paper"
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

@log_progress("plotting loss(atom_name)")
def plot_loss_atom_name(predictions, loss = "loss"):
    losses = [prediction[4][loss] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]
    
    # Scale losses according to the atom scale factor
    losses = [loss * get_scale_factor(atom_name) for atom_name, loss in zip(atom_names, losses)]
    
    # Make NMD colors
    nmd = np.array([get_mean_distance_and_std(prediction[0])[0] for prediction in predictions])
    nmd = nmd / np.max(nmd)
    colors = [plt.cm.cool(mean_distance) for mean_distance in nmd]

    # Sort everything by mean distance
    # atom_names, losses, colors = zip(*sorted(zip(atom_names, losses, colors), key=lambda x: get_mean_distance_and_std(x[0])[0]))

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

@log_progress("plotting loss(nmd)")
def plot_loss_nmd(predictions):
    losses = [prediction[4]["loss"] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]

    # Scale losses according to the atom scale factor
    losses = [loss * get_scale_factor(atom_name) for atom_name, loss in zip(atom_names, losses)]

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

@log_progress("plotting cluster hist")
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



@log_progress("plotting molecule")
def plot_molecule(predictions, sample: int):
    """
        This function plots the molecule with the predicted coordinates. It also plots the real coordinates if they are available and the difference between the predicted and real coordinates.
        Also plots neighbor atoms if available.
    """
    pass

@log_progress("plotting bond length histrogram")
def plot_bond_length_distribution(analysis_data):
    """
        Finds all bond lengths in the predictions and true positions and plots them in a histogram.
    """
    
    bond_lengths_pred = []
    bond_lengths_true = []
    
    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0,0,0])
        
        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0,0,0])
    
    for X, Y_true, Y_pred in analysis_data:        
        for bond in [DOPC_AT_MAPPING[12]]:
            from_atom = bond[0]
            to_atom = bond[1]

            from_atom_pos_pred = find_atom(Y_pred, from_atom)[1]
            to_atom_pos_pred   = find_atom(Y_pred, to_atom)[1]

            from_atom_pos_true = find_atom(Y_true, from_atom)[1]
            to_atom_pos_true   = find_atom(Y_true, to_atom)[1]

            # Calculate bond length
            bond_length_pred = np.linalg.norm(to_atom_pos_pred - from_atom_pos_pred)
            bond_length_true = np.linalg.norm(to_atom_pos_true - from_atom_pos_true)
            
            bond_lengths_pred.append(bond_length_pred)
            bond_lengths_true.append(bond_length_true)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)
    
    # Print min max and mean
    print(f"Min bond length: {np.min(bond_lengths_pred)} Å")
    print(f"Max bond length: {np.max(bond_lengths_pred)} Å")
    print(f"Mean bond length: {np.mean(bond_lengths_pred)} Å")
    
    # Do for true 
    print(f"Min bond length: {np.min(bond_lengths_true)} Å")
    print(f"Max bond length: {np.max(bond_lengths_true)} Å")
    print(f"Mean bond length: {np.mean(bond_lengths_true)} Å")
    
    
    
    # Plot histograms
    n_true, b_true, p_true = plt.hist(bond_lengths_true, bins=100, label="True", alpha=0.75, color="blue", bottom=0, density=True, histtype="stepfilled")
    n_pred, b_pred, p_pred = plt.hist(bond_lengths_pred, bins=100, label="Predicted", alpha=0.75, color="orange", bottom=0, density=True, histtype="stepfilled")

    # Plot mean of true and pred
    plt.axvline(x=np.mean(bond_lengths_true), color="blue", alpha=0.5, linestyle="--")
    plt.axvline(x=np.mean(bond_lengths_pred), color="orange", alpha=0.5, linestyle="--")

    # Add labels
    plt.title("Bond angle distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Bond length (Å)")
    plt.legend()
    
    return fig

@log_progress("plotting bond angle histrogram")
def plot_bond_angle_distribution(predictions):
    """
        Finds all bond angles in the predictions and true positions and plots them in a histogram.
    """
    
    atom_names = [name for name in DOPC_AT_NAMES if not name.startswith("H") and not name.startswith("N")]
    
    bond_angles_pred = []
    bond_angles_true = []
    
    for atom_name, X, y_true, y_pred, losses in predictions:
        
        for bond1 in DOPC_AT_MAPPING:
            for bond2 in DOPC_AT_MAPPING:
                if bond1 == bond2:
                    continue
                
                if bond1[0] == bond2[0] or bond1[0] == bond2[1] or bond1[1] == bond2[0] or bond1[1] == bond2[1]:
                    from_atom_name1 = bond1[0]
                    to_atom_name1 = bond1[1]

                    from_atom_pos_pred1 = np.array([0,0,0])  if from_atom_name1 == "N" else y_pred[atom_names.index(from_atom_name1)]
                    to_atom_name_pred1 = np.array([0,0,0])   if to_atom_name1 == "N"   else y_pred[atom_names.index(to_atom_name1)]

                    from_atom_pos_true1 = np.array([0,0,0])  if from_atom_name1 == "N" else y_true[atom_names.index(from_atom_name1)]
                    to_atom_name_true1 = np.array([0,0,0])   if to_atom_name1 == "N"   else y_true[atom_names.index(to_atom_name1)]

                    from_atom_name2 = bond2[0]
                    to_atom_name2 = bond2[1]

                    from_atom_pos_pred2 = np.array([0,0,0])  if from_atom_name2 == "N" else y_pred[atom_names.index(from_atom_name2)]
                    to_atom_name_pred2 = np.array([0,0,0])   if to_atom_name2 == "N"   else y_pred[atom_names.index(to_atom_name2)]

                    from_atom_pos_true2 = np.array([0,0,0])  if from_atom_name2 == "N" else y_true[atom_names.index(from_atom_name2)]
                    to_atom_name_true2 = np.array([0,0,0])   if to_atom_name2 == "N"   else y_true[atom_names.index(to_atom_name2)]

                    v0_pred = to_atom_name_pred1 - from_atom_pos_pred1
                    v1_pred = to_atom_name_pred2 - from_atom_pos_pred2
                    
                    v0_true = to_atom_name_true1 - from_atom_pos_true1
                    v1_true = to_atom_name_true2 - from_atom_pos_true2

                    # Calculate angle between the two bonds
                    theta_pred = np.math.atan2(np.linalg.det([v0_pred,v1_pred]),np.dot(v0_pred,v1_pred))
                    theta_true = np.math.atan2(np.linalg.det([v0_pred,v1_pred]),np.dot(v0_pred,v1_pred))

                    bond_angles_pred.append(theta_pred)
                    bond_angles_true.append(theta_true)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)
    n_true, b_true, p_true = plt.hist(bond_angles_true, bins=100, label="True", alpha=0.75, color="blue", bottom=0, density=True, histtype="stepfilled", facecolor='none', edgecolor='k', linewidth=2)
    n_pred, b_pred, p_pred = plt.hist(bond_angles_pred, bins=100, label="Predicted", alpha=0.75, color="orange", bottom=0, density=True, histtype="stepfilled")

    # Plot difference
    plt.plot(b_true[:-1], np.abs(n_true - n_pred), label="Difference", color="black", alpha=0.25, linewidth=2, linestyle="--")
    
    # Plot mean of true and pred
    plt.axvline(x=np.mean(bond_angles_true), color="blue", alpha=0.5, linestyle="--")
    plt.axvline(x=np.mean(bond_angles_pred), color="orange", alpha=0.5, linestyle="--")
    
    # Add labels
    plt.title("Bond angle distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Bond length (Å)")
    plt.legend()
    
    return fig

@log_progress("plotting bond dihedral angle histrogram")
def plot_bond_dihedral_angle_distribution(predictions):
    """
    """
    pass

@log_progress("plotting coordinates histrogram")
def plot_coordinates_distribution(predictions, atom_name: str):
    """
    """
    pass

@log_progress("plotting radial distribution")
def plot_radial_distribution_function(predictions, atom_name: str):
    """
    """
    pass

@log_progress("plotting N molecules")
def plot_N_molecules(predictions, N: int):
    """
    """
    pass