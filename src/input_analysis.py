import logging
import os
import pickle
import socket
import sys
import time

import Bio.PDB

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from library.config import Keys, config
from library.static.vector_mappings import DOPC_AT_MAPPING, DOPC_AT_BAB
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
MEMBRANE_PATH = os.path.join(DATA_PREFIX, "training", "membranes")

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

# List all membranes in the training folder
membranes = [os.path.join(MEMBRANE_PATH, membrane) for membrane in os.listdir(MEMBRANE_PATH)]

# Load the first membrane with BioPython
parser = Bio.PDB.PDBParser(QUIET=True)


bond_lengths = [[] for _ in range(len(DOPC_AT_MAPPING))]

# Remove duplicate bond angle bonds
angle_pairs = [] # List of tuples of (name, name, name)
DOPC_AT_BAB = list(set(DOPC_AT_BAB))
for i, ((a,b), (c,d)) in enumerate(DOPC_AT_BAB):
    # Find common atom
    if a == c:
        angle_pairs.append((b, d, a))
    elif a == d:
        angle_pairs.append((b, c, a))
    elif b == c:
        angle_pairs.append((a, d, b))
    elif b == d:
        angle_pairs.append((a, c, b))
    else:
        raise ValueError(f"Could not find common atom in {((a,b), (c,d))}")

bond_angles = [[] for _ in range(len(angle_pairs))]
bond_angles_stats = [[] for _ in range(len(angle_pairs))]

for k, membrane in enumerate(membranes):
    # Load the whole membrane
    structure = parser.get_structure("membrane", os.path.join(membrane, "at.pdb"))

    # Get all residues (molecules) in the membrane
    residues = structure.get_residues()

    # Loop through all residues
    for residue in residues:
        # Get all atoms in the residue
        atoms = list(residue.get_atoms())

        # Save all bond lengths in this residue
        for i,(a,b) in enumerate(DOPC_AT_MAPPING):
            atom_a = [atom for atom in atoms if atom.get_name() == a][0]
            atom_b = [atom for atom in atoms if atom.get_name() == b][0]
            bond_lengths[i].append(atom_a - atom_b)

        for i, (a,b,c) in enumerate(angle_pairs):
            atom_a = [atom for atom in atoms if atom.get_name() == a][0]
            atom_b = [atom for atom in atoms if atom.get_name() == b][0]
            atom_c = [atom for atom in atoms if atom.get_name() == c][0]
            
            # Calculate angle between the two bonds
            angle = Bio.PDB.calc_angle(atom_a.get_vector(), atom_c.get_vector(), atom_b.get_vector())
            bond_angles[i].append(angle / np.pi * 180)

    # Print progress
    if k % 10 == 0:
        print(f"Processed membrane {k}/{len(membranes)}")

    break

# Calculate the mean and standard deviation of the bond lengths for every bond
for i, bond in enumerate(bond_lengths):
    bond_lengths[i] = (np.mean(bond), np.std(bond))
    print(f"Mean bond length of {DOPC_AT_MAPPING[i]}: {bond_lengths[i][0]} +- {bond_lengths[i][1]}")

# Calculate the mean and standard deviation of the bond angles for every bond
for i, bond in enumerate(bond_angles):
    bond_angles_stats[i] = (np.mean(bond), np.std(bond))
    print(f"Mean bond angle of {angle_pairs[i]}: {bond_angles_stats[i][0]} +- {bond_angles_stats[i][1]}")
    
over_2 = 0
under_2 = 0

# Make bond angle historgram
for i, bond in enumerate(angle_pairs):
    bond_a = bond_angles[i]

    # Ignore outliers
    bond_a = np.array(bond_a)
    bond_a = bond_a[bond_a > bond_angles_stats[i][0] - 2 * bond_angles_stats[i][1]] # Ignore outliers (2 standard deviations)
    
    # Calculate mean and standard deviation
    bond_angles_stats[i] = (np.mean(bond_a), np.std(bond_a))
    
    if(bond_angles_stats[i][1] > 2):
        over_2 += 1
    else:
        under_2 += 1
        
    print(f"Number of bond angles with a standard deviation of more than 2 degrees: {over_2}")
    print(f"Number of bond angles with a standard deviation of less than 2 degrees: {under_2}")

    
    # Clear
    plt.clf()
    plt.hist(bond_a, bins=100)
    plt.xlabel(f"Bond angle (degrees) {bond} {bond_angles_stats[i][0]} +- {bond_angles_stats[i][1]}")
    plt.ylabel("Frequency")
    plt.title("Bond angle distribution")
    plt.savefig(gen_path(f"training_bond_angle_distribution_{i}.png"), **savefig_kwargs)