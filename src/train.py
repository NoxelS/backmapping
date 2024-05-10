import argparse
import os
import socket
import sys
import time

from library.config import Keys, config, print_config, set_hp_config_from_name, validate_config
from library.datagen.topology import get_ic_from_index, get_max_ic_index, ic_to_hlabel

MAX_IC_INDEX = get_max_ic_index()  # This is the maximum internal coordinate index


def train_model(target_ic_index: int, use_socket: bool = False, host_ip_address: str = "localhost", dry_run=False) -> None:
    """
    Trains a model for the given internal coordinate index.

    Args:
        target_ic_index (int): The index of the internal coordinate that should be fitted.
        use_socket (bool, optional): Whether to use a socket to communicate with the parent process. Defaults to False.
        host_ip_address (str, optional): The IP address of the host. Defaults to "localhost".
    """

    import tensorflow as tf

    from library.classes.generators import FICDataGenerator
    from library.classes.losses import CustomLoss
    from library.classes.models import IDOFNet, IDOFNet_Reduced
    from master import PORT, encode_finished, encode_starting

    ##### SOCKET #####
    client = None
    if use_socket and not dry_run:
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
    if use_socket and dry_run:
        print("Socket communication is not possible in dry run mode. Disabling socket communication.")
        use_socket = False

    ##### TRAINING #####

    INPUT_SIZE = (12 + 2 * config(Keys.PADDING), 3 * (1 + config(Keys.NEIGHBORHOOD_SIZE)) + 2 * config(Keys.PADDING), 1)
    OUTPUT_SIZE = (1 + 2 * config(Keys.PADDING), 1 + 2 * config(Keys.PADDING), 1)

    sample_gen = FICDataGenerator(
        input_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "input"),
        output_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "output"),
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        shuffle=config(Keys.SAMPLEGEN_SHUFFLE),
        batch_size=config(Keys.SAMPLEGEN_BATCH_SIZE),
        validate_split=config(Keys.VALIDATION_SPLIT),
        validation_mode=False,
        augmentation=config(Keys.SAMPLEGEN_AUGMENTATION),
        ic_index=target_ic_index,
        neighbourhood_size=config(Keys.NEIGHBORHOOD_SIZE),
        use_cache=config(Keys.SAMPLEGEN_USE_CACHE),
    )

    train_gen = FICDataGenerator(
        input_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "input"),
        output_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "output"),
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        shuffle=config(Keys.TRAINGEN_SHUFFLE),
        batch_size=config(Keys.TRAINGEN_BATCH_SIZE),
        validate_split=config(Keys.VALIDATION_SPLIT),
        validation_mode=False,
        augmentation=config(Keys.TRAINGEN_AUGMENTATION),
        ic_index=target_ic_index,
        neighbourhood_size=config(Keys.NEIGHBORHOOD_SIZE),
        data_usage=config(Keys.DATA_USAGE),
        use_cache=config(Keys.TRAINGEN_USE_CACHE),
    )

    validation_gen = FICDataGenerator(
        input_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "input"),
        output_dir_path=os.path.join(config(Keys.DATA_PATH), "training", "output"),
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        shuffle=config(Keys.VALIDGEN_SHUFFLE),
        batch_size=config(Keys.VALIDGEN_BATCH_SIZE),
        validate_split=config(Keys.VALIDATION_SPLIT),
        validation_mode=True,
        augmentation=config(Keys.VALIDGEN_AUGMENTATION),
        ic_index=target_ic_index,
        neighbourhood_size=config(Keys.NEIGHBORHOOD_SIZE),
        data_usage=config(Keys.DATA_USAGE),
        use_cache=config(Keys.VALIDGEN_USE_CACHE),
    )

    # The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
    # performance when training on multiple GPUs.
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    default_startegy = tf.distribute.get_strategy()

    print(f"Starting to load and train the model for internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)})")

    with strategy if config(Keys.USE_CENTRAL_STORAGE_STRATEGY) else default_startegy.scope():

        try:
            # Select the right network type based on the configuration
            try:
                networks = {"IDOFNet": IDOFNet, "IDOFNet_Reduced": IDOFNet_Reduced}
                target_network: IDOFNet = networks[config(Keys.NETWORK)]
            except KeyError:
                raise ValueError(f"Invalid network type: '{config(Keys.NETWORK)}'. Choose one of [{', '.join(networks.keys())}].")

            # Create the network. This will also load the model if it exists.
            net = target_network(
                INPUT_SIZE,
                OUTPUT_SIZE,
                data_prefix=config(Keys.DATA_PATH),
                display_name=f"{config(Keys.MODEL_NAME_PREFIX)}_{target_ic_index}",
                keep_checkpoints=True,
                load_path=os.path.join(config(Keys.DATA_PATH), "models", str(target_ic_index), f"{config(Keys.MODEL_NAME_PREFIX)}.h5"),
                # loss=CustomLoss(),
                test_sample=sample_gen.__getitem__(0),
                socket=client if use_socket else None,
                host_ip_address=host_ip_address if use_socket else None,
                port=PORT if use_socket else None,
                ic_index=target_ic_index,
            )

            # Abort if we are in dry run mode
            if dry_run:
                print("Dry run mode is enabled. Aborting training now. Everything seems to be set up correctly up to this point.")
                return

            # Train the model
            try:
                net.fit(
                    train_gen,
                    batch_size=config(Keys.BATCH_SIZE),
                    epochs=config(Keys.EPOCHS),
                    validation_gen=validation_gen,
                    use_tensorboard=config(Keys.USE_TENSORBOARD),
                    early_stop=config(Keys.USE_EARLY_STOP),
                )
            except Exception as e:
                print(f"Could not train model: {e}")

            # Save the model
            try:
                net.save()
            except Exception as e:
                print(f"Could not save model: {e}")

        except Exception as e:
            print(f"Could not create model: {e}")

    # Send finished signal
    if use_socket:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((host_ip_address, PORT))
            client.send(encode_finished(target_ic_index))
            client.close()
        except Exception as e:
            print(f"Could not send finished signal: {e}")


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Train a model for a specific internal coordinate.")

    # Add internal coordinate index argument
    parser.add_argument(dest="ic_index", type=int, help="The index of the free internal coordinate that should be fitted")

    # Add optional host argument
    parser.add_argument("--host", type=str, help="The IP address of the master. If not provided, the script will run in standalone mode.")

    # Add optional hyperparameter configuration name argument
    parser.add_argument("--config", type=str, help="The name of the hyperparameter configuration.", default="default")

    # Add dry run argument
    parser.add_argument("--dry-run", action="store_true", help="Whether to run the script in dry run mode.", default=False)

    # Add purge run argument
    parser.add_argument(
        "--purge", action="store_true", help="Whether to clean caches and saves of the model before training. This does not remove data generator caches!", default=False
    )

    # Add purge generator caches argument
    parser.add_argument(
        "--purge-gen-caches",
        action="store_true",
        help="Whether to clean data generator caches before training. IMPORTANT: This removed all caches that contain the target ic index because the cache files do not depend on the model structure or hyperparameters!",
        default=False,
    )

    args = parser.parse_args()

    # Check if the internal coordinate index is valid
    target_ic_index = args.ic_index
    if target_ic_index < 0 or target_ic_index > MAX_IC_INDEX:
        raise Exception(f"Invalid ic index: {target_ic_index}, choose one of: 0-{MAX_IC_INDEX}")

    # Check if the internal coordinate index is a free ic
    target_ic = get_ic_from_index(target_ic_index)
    if target_ic["fixed"]:
        raise ValueError(f"Internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)}) is not a free internal coordinate!")

    # Clean the model directory if requested
    if args.purge:
        dirs_to_clean = [
            os.path.join(config(Keys.DATA_PATH), "models", str(target_ic_index), f"{config(Keys.MODEL_NAME_PREFIX)}.h5"),
            os.path.join(config(Keys.DATA_PATH), "hist", f"training_history_{config(Keys.MODEL_NAME_PREFIX)}_{str(target_ic_index)}.csv"),
        ]

        if args.dry_run:
            print("Would clean up caches and saves:", dirs_to_clean)

        else:
            print("Cleaning up caches and saves...")
            # Linux
            for file in dirs_to_clean:
                # Try as a file
                try:
                    os.remove(file)
                except Exception as e:
                    # Linux
                    try:
                        os.system(f"rm -rf {file}")
                    except Exception as e:
                        # Windows
                        try:
                            os.system(f"rmdir /s /q {file}")
                        except Exception as e:
                            pass

    # Clean the data generator caches if requested
    # Note: This removed all caches that contain the target ic index because the cache files
    # do not depend on the model structure or hyperparameters.
    if args.purge_gen_caches:
        cache_files = [_ for _ in os.listdir(os.path.join(config(Keys.DATA_PATH), "cache")) if f"_{target_ic_index}_" in _]

        if args.dry_run:
            print("Would clean up data generator caches:", cache_files)
        else:
            print("Cleaning up data generator caches...")
            for file in cache_files:
                try:
                    os.remove(os.path.join(config(Keys.DATA_PATH), "cache", file))
                except Exception as e:
                    pass

    # Load the configuration and validate it
    set_hp_config_from_name(args.config)
    validate_config()

    # Print the configuration
    print("Successfully loaded configuration:")
    print_config()

    # Run the script
    train_model(args.ic_index, use_socket=bool(args.host), host_ip_address=args.host, dry_run=args.dry_run)
