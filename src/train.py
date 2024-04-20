import os
import socket
import sys
import time

from library.static.utils import print_progress_bar

MAX_STEPS = 7
print_progress_bar(0, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading Tensorflow...")

import tensorflow as tf

from library.classes.generators import FICDataGenerator
from library.classes.models import CNN, MODEL_TYPES
from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, get_IC_max_index, ic_to_hlabel
from master import PORT, encode_finished, encode_starting

print_progress_bar(1, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading config...")


##### CONFIGURATION #####

DATA_PREFIX = config(Keys.DATA_PATH)
BATCH_SIZE = config(Keys.BATCH_SIZE)
VALIDATION_SPLIT = config(Keys.VALIDATION_SPLIT)
NEIGHBORHOOD_SIZE = config(Keys.NEIGHBORHOOD_SIZE)
EPOCHS = config(Keys.EPOCHS)
MODEL_NAME_PREFIX = config(Keys.MODEL_NAME_PREFIX)
DATA_USAGE = config(Keys.DATA_USAGE)
USE_TENSORBOARD = config(Keys.USE_TENSORBOARD)
PADDING = int(config(Keys.PADDING))


"""
    The cg size is currently set to 12 + 12 * NEIGHBORHOOD_SIZE + 2 * PADDING, 3 + 2 * PADDING, 1:
    ------------------------------------------
    | CG , Neighbor, Neighbor, ..., Neighbor |
    ------------------------------------------
    
    TODO: make it like this:
    -------------------------------------
    |             Neighbor              |
    |             Neighbor              |
    |                CG                 |
    |             Neighbor              |
    |             Neighbor              |
    -------------------------------------
"""
CG_SIZE = (12 + 12 * NEIGHBORHOOD_SIZE + 2 * PADDING, 3 + 2 * PADDING, 1)  # Needs to be one less than the actual size for relative vectors


"""
    The at size is currently set to be 1 + 2 * PADDING, 2 * PADDING, 1:
    -----------
    | X  X  X |
    | X  IC X |
    | X  X  X |
    -----------
    
"""
OUTPUT_SIZE = (1 + 2 * PADDING, 1 + 2 * PADDING, 1)

print_progress_bar(2, MAX_STEPS, prefix="Setting up the training environment", suffix="Checking arguments...")

# This is the maximum internal coordinate index
MAX_IC_INDEX = get_IC_max_index()

# Check if the internal coordinate index is valid
if len(sys.argv) < 2:
    raise Exception(f"Please provide the index of the internal coordinate that should be fitted as an argument, choose one out of: 0-{MAX_IC_INDEX}")

# Check if the internal coordinate index is valid
target_ic_index = int(sys.argv[1])
if target_ic_index < 0 or target_ic_index > MAX_IC_INDEX:
    raise Exception(f"Invalid ic index: {target_ic_index}, choose one of: 0-{MAX_IC_INDEX}")

# Check if the internal coordinate index is a free ic
target_ic = get_ic_from_index(target_ic_index)
if target_ic["fixed"]:
    raise ValueError(f"Internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)}) is not a free internal coordinate!")


##### SOCKET #####

use_socket = len(sys.argv) > 2
client = None

if use_socket:
    print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading socket...")

    # Try to connect to the parent process
    try:
        host_ip_address = sys.argv[2]
        print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix=f"Trying to connect to parent {host_ip_address}...")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host_ip_address, PORT))
        client.send(encode_starting(target_ic_index))
        client.close()
    except Exception as _:
        # Sleep for 30 seconds to give the parent process time to start the server
        print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix="Retrying to connect to parent...")
        time.sleep(30)

        # Try again
        try:
            print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix="Retrying to connect to parent...")
            host_ip_address = sys.argv[2]
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((host_ip_address, PORT))
            client.send(encode_starting(target_ic_index))
            client.close()
        except ConnectionRefusedError:
            print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix="No parent found, not using socket...")
            use_socket = False
        except TimeoutError:
            print_progress_bar(3, MAX_STEPS, prefix="Setting up the training environment", suffix="No parent found, not using socket...")
            use_socket = False


##### TRAINING #####

print_progress_bar(4, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading sample generator...")
sample_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
)

print_progress_bar(5, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading training generator...")
train_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=True,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE,
)

print_progress_bar(6, MAX_STEPS, prefix="Setting up the training environment", suffix="Loading validation generator...")
validation_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    validate_split=VALIDATION_SPLIT,
    validation_mode=True,
    augmentation=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    data_usage=DATA_USAGE,
)


# The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
# performance when training on multiple GPUs.
strategy = tf.distribute.experimental.CentralStorageStrategy()

print_progress_bar(7, MAX_STEPS, prefix="Setting up the training environment", suffix="Finished...                                           ")
print(f"Starting to load and train the model for internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)})")

with strategy.scope():

    cnn = CNN(
        CG_SIZE,
        OUTPUT_SIZE,
        data_prefix=DATA_PREFIX,
        display_name=f"{MODEL_NAME_PREFIX}_{target_ic_index}",
        keep_checkpoints=True,
        load_path=os.path.join(DATA_PREFIX, "models", str(target_ic_index), f"{MODEL_NAME_PREFIX}.h5"),
        # We currently use the keras MeanAbsoluteError loss function, because custom loss functions are not supproted while saving the model
        # in the current tensorflow version. This hopefully will change in the future.
        loss=tf.keras.losses.MeanAbsoluteError(),
        test_sample=sample_gen.__getitem__(0),
        socket=client if use_socket else None,
        host_ip_address=host_ip_address if use_socket else None,
        port=PORT if use_socket else None,
        model_type=MODEL_TYPES.CNN_V1_1,
    )

    cnn.fit(train_gen, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_gen=validation_gen, use_tensorboard=USE_TENSORBOARD)

    try:
        cnn.save()
    except Exception as e:
        print(f"Could not save model: {e}")

# Send finished signal
if use_socket:
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host_ip_address, PORT))
        client.send(encode_finished(target_ic_index))
        client.close()
    except Exception as e:
        print(f"Could not send finished signal: {e}")
