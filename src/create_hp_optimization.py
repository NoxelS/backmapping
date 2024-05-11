import argparse
from ast import expr_context
from library.config import config, Keys, set_hp_config, set_hp_config_from_name, validate_config
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a hyperparameter optimization experiment')

    # Create a hyperparameter optimization experiment by providing a range of hyperparameters for on key in the config
    # thill will create a new experiment for each combination of hyperparameters

    # Add argument for the base hyperparameter
    parser.add_argument('--base-config', type=str, help='The name of the base config to use', default="default")
    # Add argument for the keys to optimize
    for key in [key for key in Keys if key.value >= 1000]:
        name = key.name.lower().replace("_", "-")
        # Add argument for the minimum value of the key
        parser.add_argument(
            f"--{name}-min",
            type=float,
            help=f"The minimum of values for the key {key.name}",
        )

        # Add argument for the maximum value of the key
        parser.add_argument(
            f"--{name}-max",
            type=float,
            help=f"The maximum of values for the key {key.name}",
        )

        # Add argument for the amount of steps in the range given
        parser.add_argument(
            f"--{name}-steps",
            type=float,
            help=f"The amount of steps in the range of {key.name}",
        )

    args = vars(parser.parse_args())

    # Filter out the keys that are not set
    keys = {key: args[key] for key in args if args[key] is not None}

    # Load the base config and validate it
    set_hp_config_from_name(args["base_config"])
    validate_config()

    # Remove base argument from the keys
    keys.pop("base_config")

    # Check if one of an argument for a key is set then every argument for that key should be set
    for key in keys:
        # Get the base name of the key
        base_name = key[::-1].split("_", 1)[1][::-1]
        if args[f"{base_name}_max"] is None or args[f"{base_name}_min"] is None or args[f"{base_name}_steps"] is None:
            raise Exception(f"Please provide all arguments for the key {base_name}")

    # Parse the arguments to the correct types
    for key in keys:
        keys[key] = float(keys[key])

    # Create a dict of the hyperparameter base name with the range of values as value
    hyperparameters = {}

    for key in keys:
        base_name = key[::-1].split("_", 1)[1][::-1]
        range = np.linspace(
            keys[f"{base_name}_min"],
            keys[f"{base_name}_max"],
            int(keys[f"{base_name}_steps"]),
        )
        hyperparameters[base_name] = range

        # Print a message to the user if the range includes float values
        if list(range) != list(range.astype(int)):
            print(f"The range of {base_name} includes float values")

    print(hyperparameters)
