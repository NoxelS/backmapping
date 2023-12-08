import os
import sys
import socket
import subprocess
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

RAW_DATA_PATH = "/localdisk/users/noel/DOPC_CG_2_AA_NEW.zip"

## Find the raw data zip file
if os.path.exists(RAW_DATA_PATH):
    subprocess.call(f"unzip {RAW_DATA_PATH} -d {DATA_PREFIX}")
    subprocess.call(f"mv {DATA_PREFIX}/DOPC_CG_2_AA_NEW/ {DATA_PREFIX}/raw/")