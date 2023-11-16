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
FIG_SIZE = (20, 10)

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


### Plot functions ###

def plot_loss_atom_name(predictions, loss = "loss"):
    losses = [prediction[4][loss] for prediction in predictions]
    atom_names = [prediction[0] for prediction in predictions]
    
    # Create a figure
    fig = plt.figure(figsize=FIG_SIZE)
    plt.bar(atom_names, losses)
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
    fig = plt.figure(figsize=FIG_SIZE)
    plt.scatter(nmd, losses)
    plt.title(f"Loss for each model")
    plt.ylabel("Loss")
    plt.xlabel("Normalized mean distance of model")
    
    return fig
