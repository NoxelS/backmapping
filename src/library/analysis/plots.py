import logging
import os
import pickle
import socket
import sys
import time

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from library.classes.losses import CustomLoss
from library.classes.models import IDOFNet
from library.config import Keys, config
from library.plot_config import set_plot_config
from library.static.topologies import DOPC_AT_NAMES
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP, log_progress
from library.static.vector_mappings import DOPC_AT_MAPPING

##### CONFIGURATION #####

# Analysis config
SAMPLE_SIZE = 64

# Plot config
THEME = "seaborn-paper"  # "seaborn-v0_8-paper"
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
ATOM_NAMES_TO_FIT = [
    name for name in DOPC_AT_NAMES if not name.startswith("H")
]  # Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
ATOM_NAMES_TO_FIT_WITH_MODEL = [
    name for name in ATOM_NAMES_TO_FIT if os.path.exists(os.path.join(DATA_PREFIX, "models", name, f"{MODEL_NAME_PREFIX}.h5"))
]  # Check which of those atoms already have a model

# Use python theme
plt.style.use(THEME) if THEME in plt.style.available else print(f"Theme '{THEME}' not available, using default theme. Select one of {plt.style.available}.")


### Plot functions ###


@log_progress("plotting loss(atom_name)")
def plot_loss_atom_name(predictions, loss="loss"):
    losses = [prediction[4][loss] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]

    # Scale losses according to the atom scale factor
    # TODO: fix
    # losses = [loss * get_output_scale_factor(atom_name) for atom_name, loss in zip(atom_names, losses)]

    # Make NMD colors
    nmd = 1  # TODO: fix
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

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    return fig


@log_progress("plotting loss(nmd)")
def plot_loss_nmd(predictions):
    losses = [prediction[4]["loss"] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]

    # Scale losses according to the atom scale factor
    # TODO: fix
    # losses = [loss * get_output_scale_factor(atom_name) for atom_name, loss in zip(atom_names, losses)]

    nmd = 1  # TODO: fix

    # Normalize
    nmd = nmd / np.max(nmd)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_SQUARE)
    plt.scatter(nmd, losses)
    plt.title(f"Loss for each model")
    plt.ylabel("Loss")
    plt.xlabel("Normalized mean distance of model")

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    return fig


@log_progress("plotting cluster hist")
def plot_cluster_hist(data_col=2):

    # Load plot config
    set_plot_config()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    min_loss = 99999

    hist_files = os.listdir(PATH_TO_HIST)

    # Data to find the average graph
    avg_data = []

    # Loop over all files in the hist folder
    for i, hist in enumerate(hist_files):
        # file is named: training_history_prefix_ic.csv
        ic_index = hist.split("_")[-1].split(".")[0]

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

        if np.min(hist[:, int(data_col)]) < min_loss:
            min_loss = np.min(hist[:, int(data_col)])

        # Plot
        # color = plt.cm.cool(mean_distance)
        ax.plot(hist[:, 0] + 1, hist[:, int(data_col)], label=str(ic_index))

        # Add to average data
        avg_data.append(hist[:, int(data_col)])

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
    data_name = ["", "Accuracy", "MSE Loss (Å)", "Learning Rate", "Mean Average Error", "Val. Accuracy", "Val. MSE Loss (Å)", "Val. Mean Average Error"][int(data_col)]

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(data_name)
    ax.set_title("Training History")

    # Add grid
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", alpha=0.5)

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
    ax.text(
        text_center_x, text_center_y, f"Minimum Loss: {min_loss:.2f} Å", horizontalalignment="center", verticalalignment="bottom", fontsize=12, color="black", alpha=0.75
    )

    # Plot legend outside of plot in two columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=2)
    ax.get_legend().set_title("Atom Names")

    # Add legend that explains color to the bottom
    ax2 = fig.add_axes([0.93, 0.11, 0.2, 0.05])
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation="horizontal")
    ax2.set_title("NMD")  # Normalized Mean Distance

    return fig


@log_progress("3d plotting molecule")
def plot_molecule(analysis_data, sample: int):
    """
    This function plots the molecule with the predicted coordinates. It also plots the real coordinates if they are available and the difference between the predicted and real coordinates.
    Also plots neighbor atoms if available.
    """

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for i_name, i_pos in array:
            if i_name == atom_name:
                return np.array(i_pos)

        raise Exception(f"Atom {atom_name} not found in array {array}")

    X, Y_true, Y_pred = analysis_data[sample]

    # Plot the molecule
    fig = plt.figure(figsize=FIG_SIZE_SQUARE)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the input
    for i, (atom_name, atom_pos) in enumerate(X):
        if i < 12:
            # Make size of the beads bigger
            ax.scatter(atom_pos[0], atom_pos[1], atom_pos[2], color="black", alpha=0.1, s=100, linewidth=0.0)
        else:
            ax.scatter(atom_pos[0], atom_pos[1], atom_pos[2], color="blue", alpha=0.15, s=100, linewidth=0.0)

    # Plot predicted atoms
    for atom_name, atom_pos in Y_pred:
        element = "".join([i for i in atom_name if i.isalpha()])
        ax.scatter(atom_pos[0], atom_pos[1], atom_pos[2], color=DEFAULT_ELEMENT_COLOR_MAP[element], alpha=0.75)
        # ax.text(atom_pos[0], atom_pos[1], atom_pos[2], atom_name, color=DEFAULT_ELEMENT_COLOR_MAP[element], fontsize=12)

    # Plot the true molecule
    for atom_name, atom_pos in Y_true:
        element = "".join([i for i in atom_name if i.isalpha()])
        ax.scatter(atom_pos[0], atom_pos[1], atom_pos[2], color=DEFAULT_ELEMENT_COLOR_MAP[element], alpha=0.5)
        # ax.text(atom_pos[0], atom_pos[1], atom_pos[2], atom_name, color=DEFAULT_ELEMENT_COLOR_MAP[element], fontsize=12)

    for bond in DOPC_AT_MAPPING:
        from_atom_pos_pred = find_atom(Y_pred, bond[0])
        to_atom_pos_pred = find_atom(Y_pred, bond[1])

        from_atom_pos_true = find_atom(Y_true, bond[0])
        to_atom_pos_true = find_atom(Y_true, bond[1])

        # Plot bond
        ax.plot(
            [from_atom_pos_pred[0], to_atom_pos_pred[0]],
            [from_atom_pos_pred[1], to_atom_pos_pred[1]],
            [from_atom_pos_pred[2], to_atom_pos_pred[2]],
            color="purple",
            alpha=0.75,
            linestyle="--",
            linewidth=0.5,
        )
        ax.plot(
            [from_atom_pos_true[0], to_atom_pos_true[0]],
            [from_atom_pos_true[1], to_atom_pos_true[1]],
            [from_atom_pos_true[2], to_atom_pos_true[2]],
            color="blue",
            alpha=0.5,
            linestyle="--",
            linewidth=0.5,
        )

    # Add labels
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"Molecule {sample}")
    ax.legend()

    return fig


@log_progress("plotting total bond length histrogram")
def plot_bond_length_distribution(analysis_data, bond):
    """
    Finds all bond lengths in the predictions and true positions and plots them in a histogram.

    Args:
        analysis_data (list): List of tuples of type (X, Y_true, Y_pred) where X is the input, Y_true is the true output and Y_pred is the predicted output.
        bond (tuple): Tuple of type (from_atom, to_atom) where from_atom and to_atom are the names of the atoms that form the bond.
    """
    from_atom, to_atom = bond

    bond_lengths_pred = []
    bond_lengths_true = []

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for i_name, i_pos in array:
            if i_name == atom_name:
                return np.array(i_pos)

        raise Exception(f"Atom {atom_name} not found in array {array}")

    for X, Y_true, Y_pred in analysis_data:
        from_atom = bond[0]
        to_atom = bond[1]

        from_atom_pos_pred = find_atom(Y_pred, from_atom)
        to_atom_pos_pred = find_atom(Y_pred, to_atom)

        from_atom_pos_true = find_atom(Y_true, from_atom)
        to_atom_pos_true = find_atom(Y_true, to_atom)

        # Calculate bond length
        bond_length_pred = np.linalg.norm(to_atom_pos_pred - from_atom_pos_pred)
        bond_length_true = np.linalg.norm(to_atom_pos_true - from_atom_pos_true)

        bond_lengths_pred.append(bond_length_pred)
        bond_lengths_true.append(bond_length_true)

    # Remove all bond lengths that are longer than 2 Å in the predicted data
    bond_lengths_pred = [bond_length for bond_length in bond_lengths_pred if bond_length < 2]

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Create histogram points
    bins = 150

    pred_hist, pred_bins = np.histogram(bond_lengths_pred, bins=bins, density=True)
    true_hist, true_bins = np.histogram(bond_lengths_true, bins=pred_bins, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3, mode="wrap")
        pred_hist = savgol_filter(pred_hist, 8, 3, mode="wrap")
    except Exception as e:
        print(e)

    # Remove negative values
    true_hist = np.clip(true_hist, 0, np.inf)
    pred_hist = np.clip(pred_hist, 0, np.inf)

    # Plot histograms as line graphs
    plt.plot(true_bins[:-1], true_hist, label="True", alpha=0.75, color="blue", linewidth=1.5)
    plt.plot(pred_bins[:-1], pred_hist, label="Predicted", alpha=0.75, color="purple", linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(true_bins[:-1], true_hist, pred_hist, facecolor="blue", interpolate=True, alpha=0.1, hatch=r"//", edgecolor="blue", linewidth=0.0)

    # Add small text to the bottom that state we used savgol filter
    plt.text(
        0.02,
        0.02,
        "Smoothed with SavGol filter",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        alpha=0.5,
    )

    # Only plot between predicted and true range
    plt.xlim(np.min([*pred_bins, *true_bins]), np.max([*pred_bins, *true_bins]))

    # Add labels
    plt.title(f"Bond length distribution {from_atom}-{to_atom}")
    plt.ylabel("Frequency")
    plt.xlabel("Bond length (Å)")
    plt.legend()

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    return fig


@log_progress("plotting total bond length histrogram")
def plot_total_bond_length_distribution(analysis_data, skip_atoms=None):
    """
    Finds all bond lengths in the predictions and true positions and plots them in a histogram.
    """

    bond_lengths_pred = []
    bond_lengths_true = []

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0, 0, 0])

    for X, Y_true, Y_pred in analysis_data:
        for bond in DOPC_AT_MAPPING:
            if skip_atoms and (bond[0] in skip_atoms or bond[1] in skip_atoms):
                continue

            from_atom = bond[0]
            to_atom = bond[1]

            from_atom_pos_pred = find_atom(Y_pred, from_atom)
            to_atom_pos_pred = find_atom(Y_pred, to_atom)

            from_atom_pos_true = find_atom(Y_true, from_atom)
            to_atom_pos_true = find_atom(Y_true, to_atom)

            # Calculate bond length
            bond_length_pred = np.linalg.norm(to_atom_pos_pred - from_atom_pos_pred)
            bond_length_true = np.linalg.norm(to_atom_pos_true - from_atom_pos_true)

            bond_lengths_pred.append(bond_length_pred)
            bond_lengths_true.append(bond_length_true)

    # Cut off any bond lengths that are longer than 2 Å in the predicted data
    bond_lengths_pred = [bond_length for bond_length in bond_lengths_pred if bond_length < 2]

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Make histogram points
    bins = 500

    hist_pred, bins_pred = np.histogram(bond_lengths_pred, bins=bins, density=True)
    hist_true, bins_true = np.histogram(bond_lengths_true, bins=bins_pred, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3)
        pred_hist = savgol_filter(pred_hist, 8, 3)
    except Exception as e:
        print(e)

    # Plot histograms as line graphs
    plt.plot(bins_pred[:-1], hist_true, label="True", alpha=0.75, color="blue", linewidth=1.5)
    plt.plot(bins_pred[:-1], hist_pred, label="Predicted", alpha=0.75, color="orange", linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(bins_pred[:-1], hist_true, hist_pred, facecolor="blue", interpolate=True, alpha=0.1, hatch=r"//", edgecolor="blue", linewidth=0.0)

    # Plot in range of true bond lengths
    plt.xlim(0, 2)

    # More ticks
    plt.xticks(np.linspace(0, 2, 20))

    # Add labels
    plt.title("Bond length distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Bond length (Å)")
    plt.legend()

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    # Add disclaimer if atoms were skipped
    if skip_atoms:
        plt.text(
            0.02,
            0.02,
            f"Skipped atoms: {skip_atoms}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=plt.gca().transAxes,
            fontsize=10,
            color="black",
            alpha=0.5,
        )

    return fig


@log_progress("plotting bond dihedral angle histrogram")
def plot_bond_dihedral_angle_distribution(predictions):
    """ """
    pass


@log_progress("plotting coordinates histrogram")
def plot_coordinates_distribution(analysis_data, atom_name: str, coordinate: str):
    """
    Plots the distribution of the coordinates for a given atom. x,y,z are plotted as histograms in the same plot.
    """
    c = ["x", "y", "z"].index(coordinate.lower())  # Get index of coordinate

    c_pred = []
    c_true = []

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0, 0, 0])

    for X, Y_true, Y_pred in analysis_data:
        true = find_atom(Y_true, atom_name)
        pred = find_atom(Y_pred, atom_name)
        c_pred.append(pred[c])
        c_true.append(true[c])

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Make relative to the mean
    c_true_mean = np.mean(c_true)

    c_pred = np.array(c_pred) - c_true_mean
    c_true = np.array(c_true) - c_true_mean

    # Calculate hisotgram points to plot it as line graph
    bins = 150

    c_true_hist, c_true_bins = np.histogram(c_true, bins=bins, density=True)
    c_pred_hist, _ = np.histogram(c_pred, bins=c_true_bins, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3)
        pred_hist = savgol_filter(pred_hist, 8, 3)
    except Exception as e:
        print(e)

    color_true = ["red", "green", "blue"][c]
    color_pred = ["violet", "lime", "cyan"][c]

    # Plot histograms
    plt.plot(c_true_bins[:-1], c_true_hist, label="True", alpha=0.75, color=color_true, linewidth=1.5)
    plt.plot(c_true_bins[:-1], c_pred_hist, label="Predicted", alpha=0.75, color=color_pred, linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(c_true_bins[:-1], c_true_hist, c_pred_hist, facecolor=color_true, interpolate=True, alpha=0.1, hatch=r"//", edgecolor=color_true, linewidth=0.0)

    # Make axis in the middle
    plt.axvline(x=0, color="k")

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    # Add labels
    plt.title(f"{coordinate.upper()}-Coordinate distribution for {atom_name}")
    plt.ylabel("Frequency")
    plt.xlabel("Coordinate (Å)")
    plt.legend()

    # Add small text to the bottom that state we used savgol filter
    plt.text(
        0.02,
        0.02,
        "Smoothed with SavGol filter",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        alpha=0.5,
    )

    return fig


@log_progress("plotting radial distribution")
def plot_radial_distribution_function(analysis_data, atom_name: str):
    """
    Plots the radial distribution function for a given atom.
    """

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0, 0, 0])

    distances_true = []
    distances_pred = []

    for X, Y_true, Y_pred in analysis_data:
        target_atom_pos_true = find_atom(Y_true, atom_name)
        target_atom_pos_pred = find_atom(Y_pred, atom_name)

        # Remove the atom we want to plot
        Y_true = np.array([i[1] for i in Y_true if i[0] != atom_name])
        Y_pred = np.array([i[1] for i in Y_pred if i[0] != atom_name])

        # Calculate distances
        [distances_true.append(np.linalg.norm(i - target_atom_pos_true)) for i in Y_true]
        [distances_pred.append(np.linalg.norm(i - target_atom_pos_pred)) for i in Y_pred]

    # Create the histogram points
    bins = 500

    hist_pred, bins_pred = np.histogram(distances_pred, bins=bins, density=True)
    hist_true, _ = np.histogram(distances_true, bins=bins_pred, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3)
        pred_hist = savgol_filter(pred_hist, 8, 3)
    except Exception as e:
        print(e)

    # Clip negative values (can happen when using savgol filter)
    hist_true = np.clip(hist_true, 0, np.inf)
    hist_pred = np.clip(hist_pred, 0, np.inf)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Plot histograms as line graphs
    plt.plot(bins_pred[:-1], hist_true, label="True", alpha=0.75, color="blue", linewidth=1.5)
    plt.plot(bins_pred[:-1], hist_pred, label="Predicted", alpha=0.75, color="purple", linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(bins_pred[:-1], hist_true, hist_pred, facecolor="blue", interpolate=True, alpha=0.1, hatch=r"//", edgecolor="blue", linewidth=0.0)

    # Add small text to the bottom that state we used savgol filter
    plt.text(
        0.02,
        0.02,
        "Smoothed with SavGol filter",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        alpha=0.5,
    )

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    # Only show the maximum range of the two bins
    plt.xlim(0, np.max([bins_pred]))

    # Add labels
    plt.title(f"Radial distribution function $g(r)$ for {atom_name}")
    plt.ylabel("g(r)")
    plt.xlabel("Distance (Å)")
    plt.legend()

    return fig


@log_progress("plotting N molecules")
def plot_N_molecules(predictions, N: int):
    """ """
    pass


@log_progress("plotting total bond angle histrogram")
def plot_bond_angle_distribution(analysis_data, bond1, bond2):
    """
    Finds all bond angles in the predictions and true positions and plots them in a histogram.

    Args:
        analysis_data (list): List of tuples of type (X, Y_true, Y_pred) where X is the input, Y_true is the true output and Y_pred is the predicted output.
        bond1 (tuple): Tuple of type (from_atom, to_atom) where from_atom and to_atom are the names of the atoms that form the bond.
        bond2 (tuple): Tuple of type (from_atom, to_atom) where from_atom and to_atom are the names of the atoms that form the bond.
    """

    # Find the common atom
    common_atom = set(bond1) & set(bond2)

    # Swap bonds so that the common atom is always the first one
    if bond1[1] == common_atom:
        bond1 = (bond1[1], bond1[0])

    if bond2[1] == common_atom:
        bond2 = (bond2[1], bond2[0])

    bond_angles_pred = []
    bond_angles_true = []

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0, 0, 0])

    for X, Y_true, Y_pred in analysis_data:
        # Find bond vectors
        bond1_vector_pred = find_atom(Y_pred, bond1[1]) - find_atom(Y_pred, bond1[0])
        bond2_vector_pred = find_atom(Y_pred, bond2[1]) - find_atom(Y_pred, bond2[0])

        bond1_vector_true = find_atom(Y_true, bond1[1]) - find_atom(Y_true, bond1[0])
        bond2_vector_true = find_atom(Y_true, bond2[1]) - find_atom(Y_true, bond2[0])

        # Calculate bond angle
        bond_angle_true = np.arccos(np.dot(bond1_vector_true, bond2_vector_true) / (np.linalg.norm(bond1_vector_true) * np.linalg.norm(bond2_vector_true)))
        bond_angle_pred = np.arccos(np.dot(bond1_vector_pred, bond2_vector_pred) / (np.linalg.norm(bond1_vector_pred) * np.linalg.norm(bond2_vector_pred)))

        bond_angles_true.append(bond_angle_true / np.pi * 180)
        bond_angles_pred.append(bond_angle_pred / np.pi * 180)

    # Make both relative to the true mean
    bond_angles_true_mean = np.mean(bond_angles_true)
    bond_angles_pred = np.array(bond_angles_pred) - bond_angles_true_mean
    bond_angles_true = np.array(bond_angles_true) - bond_angles_true_mean

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Create histogram points
    bins = 75

    true_hist, true_bins = np.histogram(bond_angles_true, bins=bins, density=True)
    pred_hist, _ = np.histogram(bond_angles_pred, bins=true_bins, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3)
        pred_hist = savgol_filter(pred_hist, 8, 3)
    except Exception as e:
        print(e)

    # Plot histograms as line graphs
    plt.plot(true_bins[:-1], true_hist, label="True", alpha=0.75, color="blue", linewidth=1.5)
    plt.plot(true_bins[:-1], pred_hist, label="Predicted", alpha=0.75, color="purple", linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(true_bins[:-1], true_hist, pred_hist, facecolor="blue", interpolate=True, alpha=0.1, hatch=r"//", edgecolor="blue", linewidth=0.0)

    # Plot mean of true and pred
    plt.axvline(x=np.mean(bond_angles_true), color="k", label="Mean True")
    plt.axvline(x=np.mean(bond_angles_pred), color="purple", label="Mean Pred")

    # Adjust x-axis to only show the range of the true bond angles
    plt.xlim(np.min(true_bins), np.max(true_bins))

    # Add small text to the bottom that state we used savgol filter
    plt.text(
        0.02,
        0.02,
        "Smoothed with SavGol filter",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        alpha=0.5,
    )

    # Change xticks labels so that they are relative to the true mean
    plt.xticks(ticks=plt.xticks()[0], labels=[f"{i:.2f}" for i in plt.xticks()[0] + bond_angles_true_mean])

    # Add labels
    plt.title(f"Bond angle distribution {bond1[1]}-{bond1[0]}-{bond2[1]}")
    plt.ylabel("Frequency")
    plt.xlabel("Bond angle (°)")
    plt.legend()

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    return fig


@log_progress("plotting total bond angle histrogram")
def plot_total_angle_distribution(analysis_data, bond_pairs):
    """
    Finds all bond lengths in the predictions and true positions and plots them in a histogram.
    """

    def find_atom(array, atom_name):
        if atom_name == "N":
            return np.array([0, 0, 0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0, 0, 0])

    # List to store all bond angles
    bond_angles_pred = []
    bond_angles_true = []

    for bond1, bond2 in bond_pairs:
        # Find the common atom
        common_atom = set(bond1) & set(bond2)

        # Swap bonds so that the common atom is always the first one
        if bond1[1] == common_atom:
            bond1 = (bond1[1], bond1[0])

        if bond2[1] == common_atom:
            bond2 = (bond2[1], bond2[0])

        for X, Y_true, Y_pred in analysis_data:
            # Find bond vectors
            bond1_vector_pred = find_atom(Y_pred, bond1[1]) - find_atom(Y_pred, bond1[0])
            bond2_vector_pred = find_atom(Y_pred, bond2[1]) - find_atom(Y_pred, bond2[0])

            bond1_vector_true = find_atom(Y_true, bond1[1]) - find_atom(Y_true, bond1[0])
            bond2_vector_true = find_atom(Y_true, bond2[1]) - find_atom(Y_true, bond2[0])

            # Calculate bond angle
            bond_angle_true = np.arccos(np.dot(bond1_vector_true, bond2_vector_true) / (np.linalg.norm(bond1_vector_true) * np.linalg.norm(bond2_vector_true)))
            bond_angle_pred = np.arccos(np.dot(bond1_vector_pred, bond2_vector_pred) / (np.linalg.norm(bond1_vector_pred) * np.linalg.norm(bond2_vector_pred)))

            bond_angles_true.append(bond_angle_true / np.pi * 180)
            bond_angles_pred.append(bond_angle_pred / np.pi * 180)

    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE_RECT)

    # Create histogram points
    bins = 200

    true_hist, true_bins = np.histogram(bond_angles_true, bins=bins, density=True)
    pred_hist, _ = np.histogram(bond_angles_pred, bins=true_bins, density=True)

    # Smooth the histogram with a savgol filter
    # Further information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    try:
        true_hist = savgol_filter(true_hist, 8, 3)
        pred_hist = savgol_filter(pred_hist, 8, 3)
    except Exception as e:
        print(e)

    # Plot histograms as line graphs
    plt.plot(true_bins[:-1], true_hist, label="True", alpha=0.75, color="blue", linewidth=1.5)
    plt.plot(true_bins[:-1], pred_hist, label="Predicted", alpha=0.75, color="purple", linewidth=1.5)

    # Fill between the two lines to show the difference
    plt.fill_between(true_bins[:-1], true_hist, pred_hist, facecolor="blue", interpolate=True, alpha=0.1, hatch=r"//", edgecolor="blue", linewidth=0.0)

    # Add small text to the bottom that state we used savgol filter
    plt.text(
        0.02,
        0.02,
        "Smoothed with SavGol filter",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        alpha=0.5,
    )

    # Add labels
    plt.title(f"Total bond angle distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Bond angle (°)")
    plt.legend()

    # Only show angles in true range
    plt.xlim(np.min(true_bins), np.max(true_bins))

    plt.xticks(np.linspace(np.min(true_bins), np.max(true_bins), 20))

    # Add grid
    plt.grid(axis="y", alpha=0.5)
    plt.grid(axis="x", alpha=0.5)

    return fig


@log_progress("transforming molecule into pdb file")
def molecule_to_ptb(analysis_data, sample_index, save_path_true, save_path_pred, save_path_CG):
    """
    This function transforms the molecule into a pdb file.
    """
    # TODO: remove this:
    # Sort by loss
    analysis_data = sorted(analysis_data, key=(lambda x: np.mean(np.linalg.norm(np.array(x[1][:, 1]) - np.array(x[2][:, 1])))))

    X, Y_true, Y_pred = analysis_data[sample_index]

    for i, array in enumerate([Y_true, Y_pred, X]):
        # Create a new structure
        structure = Structure.Structure("DOPC")

        # Create a new model
        model = Model.Model(0)
        structure.add(model)

        # Create a new chain
        chain = Chain.Chain("A")
        model.add(chain)

        # Create a new residue
        residue = Residue.Residue((" ", 1, " "), "R", " ")
        chain.add(residue)

        # Add the N origin atom (only in at-positions)
        if i == 0 or i == 1:
            atom = Atom.Atom("N", np.array([0, 0, 0]), 1.0, 1.0, " ", "N", 0, "N")
            residue.add(atom)

        # Add all atoms
        for j, (atom_name, atom_pos) in enumerate(array):
            element = "".join([j for j in atom_name if j.isalpha()])
            if i == 2 and j >= 12:
                # TODO: also show neighbor atoms, currenlty we hide them
                continue

            atom = Atom.Atom(atom_name, atom_pos, 1.0, 1.0, " ", element, 0, atom_name)
            residue.add(atom)

        # Save the structure
        io = PDBIO()
        io.set_structure(structure)
        io.save([save_path_true, save_path_pred, save_path_CG][i])
