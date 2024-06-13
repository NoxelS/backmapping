import argparse
import logging
import os
import shutil
import socket
import sys
import time
import traceback

from library.config import Keys, config, print_config, set_hp_config_from_name, validate_config
from library.datagen.topology import get_ic_from_index, get_ic_type_from_index, get_max_ic_index, ic_to_hlabel
from library.handlers import error_handled
from library.notify import send_notification

MAX_IC_INDEX = get_max_ic_index()  # This is the maximum internal coordinate index

os.environ["SM_FRAMEWORK"] = "tf.keras"


@error_handled()
def run_predictions(target_ic_index: int, dry_run=False) -> None:
    """
    Analyze the results of the training process for a specific internal coordinate.

    Args:
        target_ic_index (int): The index of the internal coordinate that should be fitted.
        use_socket (bool, optional): Whether to use a socket to communicate with the parent process. Defaults to False.
        host_ip_address (str, optional): The IP address of the host. Defaults to "localhost".
    """

    # Import tensorflow here to avoid tensorflow logging messages before the logger is set up and
    # to avoid tensorflow taking too long to import when running the script in dry run mode.
    import tensorflow as tf

    # Same for the other imports because they depend on tensorflow
    from library.classes.generators import FICDataGenerator
    from library.classes.losses import CustomLoss
    from library.classes.models import IDOFAngleNet, IDOFAngleNet_Reduced, IDOFNet, IDOFNet_Reduced

    # Define the input and output size of the model, this can be changed via the hyperparameter configuration
    # The input and output size does not include batch_sizes, those will be added on runtime by tf
    INPUT_SIZE = (12, 3 * (1 + config(Keys.NEIGHBORHOOD_SIZE)), 1)  # (cg_beads, 3(1 + N_B), 1)
    OUTPUT_SIZE = (1, 1, 1) if get_ic_type_from_index(target_ic_index) == "bond" else (1, 2, 1)

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

    # Set up the strategy for multi-GPU training, this is the default strategy
    strategy = tf.distribute.get_strategy()

    # The central storage strategy is used to synchronize the weights of the model across all GPUs. This can lead to better
    # performance when training on multiple GPUs.
    if config(Keys.USE_CENTRAL_STORAGE_STRATEGY):
        strategy = tf.distribute.experimental.CentralStorageStrategy()

    logging.debug(f"Starting to load and train the model for internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)})")

    with strategy.scope():

        try:
            # TODO: Make a static factory function in the IDOFNet class that selects the right network based on the configuration so
            #       the type of network does not have to be selected manually here.
            try:
                # Select the right network type based on the configuration
                networks = {"IDOFNet": IDOFNet, "IDOFNet_Reduced": IDOFNet_Reduced, "IDOFAngleNet_Reduced": IDOFAngleNet_Reduced, "IDOFAngleNet": IDOFAngleNet}
                target_network: IDOFNet = networks[config(Keys.NETWORK)]
            except KeyError:
                raise ValueError(f"Invalid network type: '{config(Keys.NETWORK)}'. Choose one of [{', '.join(networks.keys())}].")
            except Exception as e:
                raise ValueError(f"Could not load network: {e}")

            # Create the network. This will also load the model if it exists.
            net: IDOFNet = target_network(
                INPUT_SIZE,
                OUTPUT_SIZE,
                data_prefix=config(Keys.DATA_PATH),
                display_name=f"{config(Keys.MODEL_NAME_PREFIX)}_{target_ic_index}",
                keep_checkpoints=True,
                load_path=os.path.join(config(Keys.DATA_PATH), "models", str(target_ic_index), f"{config(Keys.MODEL_NAME_PREFIX)}.h5"),
                loss=config(Keys.LOSS_FUNCTION),  # If this is a custom loss function, this must be handled differently
                test_sample=sample_gen.__getitem__(0),
                socket=None,
                host_ip_address=None,
                port=None,
                ic_index=target_ic_index,
            )

            # Check if the model was laoded from a file
            assert net.loaded_model

            # Abort if we are in dry run mode
            if dry_run:
                logging.critical("Dry run mode is enabled. Aborting training now. Everything seems to be set up correctly up to this point.")
                return

            # Run the predictions
            train_predictions, val_predictions = net.predict_generators(train_gen, validation_gen)

            logging.info(f"Predicted {len(train_predictions)} training samples and {len(val_predictions)} validation samples.")

        except Exception as e:
            logging.error(f"Could not create model: {e}")
            print(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Train a model for a specific internal coordinate.")

    # Add internal coordinate index argument
    parser.add_argument(dest="ic_index", type=int, help="The index of the free internal coordinate that should be fitted")

    # Add optional hyperparameter configuration name argument
    parser.add_argument("-c", "--config", type=str, help="The name of the hyperparameter configuration.", default="default")

    # Add dry run argument
    parser.add_argument("--dry-run", action="store_true", help="Whether to run the script in dry run mode.", default=False)

    # Add argument for verbose
    parser.add_argument("-v", "--verbose", action="store_true", help="Turn on verbose output, defaults to off.", default=0)

    # Parse the arguments
    args = parser.parse_args()

    # Set up logger with the right verbosity
    verbosity = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(asctime)s] %(levelname)s {%(name)s:%(filename)s:%(lineno)d}: %(message)s", datefmt="%m-%d %H:%M:%S", level=verbosity)
    logging.debug("Verbose output enabled.") if args.verbose else None

    # Disable matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

    # Check if the internal coordinate index is valid
    target_ic_index = args.ic_index
    if target_ic_index < 0 or target_ic_index > MAX_IC_INDEX:
        raise Exception(f"Invalid ic index: {target_ic_index}, choose one of: 0-{MAX_IC_INDEX}")

    # Check if the internal coordinate index is a free ic
    target_ic = get_ic_from_index(target_ic_index)
    if target_ic["fixed"]:
        raise ValueError(f"Internal coordinate {target_ic_index} ({ic_to_hlabel(target_ic)}) is not a free internal coordinate!")

    # Load the configuration and validate it
    set_hp_config_from_name(args.config)
    validate_config()

    # Print the configuration
    logging.debug("Successfully loaded configuration:")
    print_config()

    # Display the hyperparameter configuration name
    logging.info(
        f"Using hyperparameter configuration '{config(Keys.HP_NAME)}' (v{config(Keys.HP_VERSION)}, {config(Keys.HP_AUTHOR)}): {config(Keys.HP_DESCRIPTION)} {config(Keys.HP_NOTES)}"
    )

    # Run the script
    run_predictions(args.ic_index, dry_run=args.dry_run)
