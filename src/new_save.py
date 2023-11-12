import os
import sys
import time
import socket

import tensorflow as tf

from master import PORT, encode_finished, encode_starting
from library.config import Keys, config
from library.classes.losses import BackmappingAbsolutePositionLoss
from library.classes.models import CNN
from library.static.topologies import DOPC_AT_NAMES
from library.classes.generators import PADDING_X, PADDING_Y, AbsolutePositionsNeigbourhoodGenerator

##### CONFIGURATION #####

DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)

cg_size = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)  # Needs to be one less than the actual size for relative vectors
at_size = (1 + 2 * int(PADDING_X), 3 + 2 * int(PADDING_Y), 1)

# Those are the atom names that are used for the training, currently all 54 atoms (without hydrogen) will be used
atom_names_to_fit = [name for name in DOPC_AT_NAMES if not name.startswith("H")]

if len(sys.argv) < 2:
    raise Exception(f"Please provide the name of the atom that should be fitted as an argument, choose one of: {', '.join(atom_names_to_fit)}")

atom_name_to_fit = sys.argv[1]

if atom_name_to_fit not in atom_names_to_fit:
    raise Exception(f"Invalid atom name: {atom_name_to_fit}, choose one of: {', '.join(atom_names_to_fit)}")

##### Transfer to new save format #####

cnn = CNN(
    cg_size,
    at_size,
    data_prefix=DATA_PREFIX,
    display_name=atom_name_to_fit,
    keep_checkpoints=True,
    load_path=os.path.join(DATA_PREFIX, "models_h5", f"{MODEL_NAME_PREFIX}{atom_name_to_fit}.h5"),
    loss=BackmappingAbsolutePositionLoss()
)

# self.model.save(os.path.join(DATA_PREFIX, "models", atom_name_to_fit, MODEL_NAME_PREFIX), overwrite=True, save_format="tf")
cnn.model.save(os.path.join(DATA_PREFIX, "models", atom_name_to_fit, f"{MODEL_NAME_PREFIX}.h5"), overwrite=True, save_format="h5")

# tf.keras.saving.save_model(cnn.model, "data/models", overwrite=True, save_format="tf")


# try:
#     cnn.save()
# except Exception as e:
#     print(f"Could not save model: {e}")
