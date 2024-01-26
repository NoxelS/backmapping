import logging
import os
import pickle
import socket
import sys
import time

from scipy.signal import savgol_filter

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure
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


### Functions ###

def bond_length_distribution(analysis_data, bond):
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
            return np.array([0,0,0])

        for i_name, i_pos in array:
            if i_name == atom_name:
                return np.array(i_pos)

        raise Exception(f"Atom {atom_name} not found in array {array}")

    for X, Y_true, Y_pred in analysis_data:
        from_atom = bond[0]
        to_atom = bond[1]

        from_atom_pos_pred = find_atom(Y_pred, from_atom)
        to_atom_pos_pred   = find_atom(Y_pred, to_atom)

        from_atom_pos_true = find_atom(Y_true, from_atom)
        to_atom_pos_true   = find_atom(Y_true, to_atom)

        # Calculate bond length
        bond_length_pred = np.linalg.norm(to_atom_pos_pred - from_atom_pos_pred)
        bond_length_true = np.linalg.norm(to_atom_pos_true - from_atom_pos_true)

        bond_lengths_pred.append(bond_length_pred)
        bond_lengths_true.append(bond_length_true)

    # Calculate mean and std 
    mean_pred = np.mean(bond_lengths_pred)
    mean_true = np.mean(bond_lengths_true)
    
    std_pred = np.std(bond_lengths_pred)
    std_true = np.std(bond_lengths_true)

    return std_true, mean_true


def bond_angle_distribution(analysis_data, bond1, bond2):
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
            return np.array([0,0,0])

        for atom in array:
            if atom[0] == atom_name:
                return atom[1]

        return np.array([0,0,0])

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

    # Calculate mean and std 
    mean_pred = np.mean(bond_angles_pred)
    mean_true = np.mean(bond_angles_true)

    std_pred = np.std(bond_angles_pred)
    std_true = np.std(bond_angles_true)
    
    return std_true, mean_true