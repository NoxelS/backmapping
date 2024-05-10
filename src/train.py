import os
import socket
import sys
import time

import tensorflow as tf

from library.classes.generators import FICDataGenerator
from library.classes.losses import CustomLoss
from library.classes.models import IDOFNet, IDOFNet_Reduced
from library.config import Keys, config, print_config
from library.datagen.topology import get_ic_from_index, get_max_ic_index, ic_to_hlabel
from master import PORT, encode_finished, encode_starting

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


# TODO: somehow manage this inside utils or something
CG_SIZE = (
    12 + 2 * PADDING,
    3 * (1 + NEIGHBORHOOD_SIZE) + 2 * PADDING,
    1,
)  # 12 because we have an origin set

OUTPUT_SIZE = (1 + 2 * PADDING, 1 + 2 * PADDING, 1)

# This is the maximum internal coordinate index
MAX_IC_INDEX = get_max_ic_index()

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
    # Try to connect to the parent process
    try:
        host_ip_address = sys.argv[2]
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host_ip_address, PORT))
        client.send(encode_starting(target_ic_index))
        client.close()

    except Exception as _:
        # Sleep for 30 seconds to give the parent process time to start the server
        time.sleep(30)

        # Try again
        try:
            host_ip_address = sys.argv[2]
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((host_ip_address, PORT))
            client.send(encode_starting(target_ic_index))
            client.close()
        except ConnectionRefusedError:
            use_socket = False
        except TimeoutError:
            use_socket = False


##### TRAINING #####

sample_gen = FICDataGenerator(
    input_dir_path=os.path.join(DATA_PREFIX, "training", "input"),
    output_dir_path=os.path.join(DATA_PREFIX, "training", "output"),
    input_size=CG_SIZE,
    output_size=OUTPUT_SIZE,
    shuffle=False,
    batch_size=1,
    validate_split=VALIDATION_SPLIT,
    validation_mode=False,
    augmentation=False,
    ic_index=target_ic_index,
    neighbourhood_size=NEIGHBORHOOD_SIZE,
    use_cache=False,
)


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

print(f"Starting to load and train the model for internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)})")

with strategy.scope():

    net = IDOFNet_Reduced(
        CG_SIZE,
        OUTPUT_SIZE,
        data_prefix=DATA_PREFIX,
        display_name=f"{MODEL_NAME_PREFIX}_{target_ic_index}",
        keep_checkpoints=True,
        load_path=os.path.join(DATA_PREFIX, "models", str(target_ic_index), f"{MODEL_NAME_PREFIX}.h5"),
        # loss=CustomLoss(),
        test_sample=sample_gen.__getitem__(0),
        socket=client if use_socket else None,
        host_ip_address=host_ip_address if use_socket else None,
        port=PORT if use_socket else None,
        ic_index=target_ic_index,
    )

    net.fit(train_gen, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_gen=validation_gen, use_tensorboard=USE_TENSORBOARD)

    try:
        net.save()
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
