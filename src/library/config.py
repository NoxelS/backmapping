import configparser
import logging
import os
from enum import Enum
from typing import Union

# The global config is used to store all global variables that are used in the overall training and analysis process.
# This includes paths to data, email parameters, master resolver parameters and global configuration parameters that
# are not part of hyperparameter optimization.
GLOBAL_CONFIG_FILE_PATH = "config.ini"
GLOBAL_CONFIG_SECTIONS = [_.upper() for _ in ["global", "train_master", "email", "data", "computational", "ntfy"]]  # Every section can hold up to 100 keys

# The hyperparameter optimization config is used to store all hyperparameters that are used in the training process.
HP_CONFIGS_PATH = os.path.join("data", "configs")
HP_DEFAULT_CONFIG_FILE_PATH = os.path.join(HP_CONFIGS_PATH, "default.ini")
HP_CONFIG_SECTIONS = [_.upper() for _ in ["data_generator", "training", "network_structure", "info", "plotting"]]  # Every section can hold up to 100 keys

# This is the current hyperparameter configuration that is used in the training process. This is set by any scripts that
# are using the hyperparameter optimization.
global __current_hp_config_path
__current_hp_config_path = HP_DEFAULT_CONFIG_FILE_PATH

# Used to cache config values for faster access
global __global_config_cache
global __hp_config_cache
__global_config_cache = {}
__hp_config_cache = {}


class Keys(Enum):
    """
    This is the global configuration key enum. It is used to access global and hyperparameter configuration parameters that are used in the
    overall training and analysis process.
    """

    """GLOBAL CONFIG"""
    # Global config
    DATA_PATH = 0

    # Master resolver config
    PORT = 100

    # Email config
    EMAIL_SERVER = 200
    EMAIL_TARGET = 201
    EMAIL_USER = 202

    # Data config
    VALIDATION_SPLIT = 301
    DATA_USAGE = 305
    MAX_TRAINING_DATA = 307

    # Computational config
    USE_CENTRAL_STORAGE_STRATEGY = 400

    # NTFY config
    USE_NTFY = 500
    NTFY_CHANNEL = 501
    NTFY_TRAINING_COOLDOWN = 502

    """HYPERPARAMETER CONFIG"""
    # Info config
    HP_NAME = 1300
    HP_DESCRIPTION = 1301
    HP_VERSION = 1302
    HP_AUTHOR = 1303
    HP_DATE = 1304
    HP_NOTES = 1305
    HP_TAGS = 1306
    MODEL_NAME_PREFIX = 1307

    # Data_generator config
    SAMPLEGEN_BATCH_SIZE = 1000
    SAMPLEGEN_SHUFFLE = 1001
    SAMPLEGEN_AUGMENTATION = 1002
    SAMPLEGEN_USE_CACHE = 1003

    TRAINGEN_BATCH_SIZE = 1004
    TRAINGEN_SHUFFLE = 1005
    TRAINGEN_AUGMENTATION = 1006
    TRAINGEN_USE_CACHE = 1007

    VALIDGEN_BATCH_SIZE = 1008
    VALIDGEN_SHUFFLE = 1009
    VALIDGEN_AUGMENTATION = 1010
    VALIDGEN_USE_CACHE = 1011

    MAX_AUGMENTATION_ANGLE = 1012

    # Training config
    BATCH_SIZE = 1100
    EPOCHS = 1101
    INITIAL_LEARNING_RATE = 1102
    USE_EARLY_STOP = 1103
    EARLY_STOP_PATIENCE = 1104
    USE_TENSORBOARD = 1105
    LR_SCHEDULER_MONITOR = 1106
    LR_SCHEDULER_FACTOR = 1107
    LR_SCHEDULER_PATIENCE = 1108
    LR_SCHEDULER_MIN_DELTA = 1109
    LR_SCHEDULER_COOLDOWN = 1110
    LR_SCHEDULER_MIN_LR = 1111
    LR_SCHEDULER_MODE = 1112

    # Network_structure config
    NETWORK = 1200
    INPUT_SCALE = 1202
    OUTPUT_SCALE = 1203
    PBC_CUTOFF = 1204
    NEIGHBORHOOD_SIZE = 1205
    LOSS_FUNCTION = 1206
    OUTPUT_ACTIVATION_FUNCTION = 1207
    FILTERS_SCALE = 1208
    USE_RELATIVE_DISTANCE_SCALE = 1209

    # Plotting config
    PREDICTION_COOLDOWN = 1400


def set_hp_config(config_path: str) -> configparser.ConfigParser:
    """
    Set the current hyperparameter config file.

    Args:
        config_path (str): The path to the hyperparameter config file.

    Returns:
        configparser.ConfigParser: The hyperparameter config parser.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    # Check if the config file is a valid config file
    hp_config = _get_config(config_path)

    # Set the current hp config path
    global __current_hp_config_path
    __current_hp_config_path = config_path

    # Clear the cache
    global __hp_config_cache
    __hp_config_cache = {}

    return hp_config


def set_hp_config_from_name(config_name: str) -> configparser.ConfigParser:
    """
    Set the current hyperparameter config file.

    Args:
        config_name (str): The name of the hyperparameter config file in the HP_CONFIGS_PATH.

    Returns:
        configparser.ConfigParser: The hyperparameter config parser.
    """
    return set_hp_config(os.path.join(HP_CONFIGS_PATH, f"{config_name}.ini"))


def get_current_hp_config_path() -> str:
    """
    Get the current hyperparameter config file path.

    Returns:
        str: The path to the current hyperparameter config file.
    """
    global __current_hp_config_path
    return __current_hp_config_path


def get_available_hp_configs() -> list:
    """
    Get a list of available hyperparameter config file names.

    Returns:
        list: A list of available hyperparameter config file names.
    """
    return [_.replace(".ini", "") for _ in os.listdir(HP_CONFIGS_PATH) if _.endswith(".ini")]


def _removeInlineComments(cfgparser):
    for section in cfgparser.sections():
        for item in cfgparser.items(section):
            cfgparser.set(section, item[0], item[1].split(";")[0].strip())


def _get_config(path=GLOBAL_CONFIG_FILE_PATH) -> configparser.ConfigParser:
    config_ = configparser.ConfigParser()
    config_.read(filenames=path, encoding="utf-8")
    _removeInlineComments(config_)
    return config_


def validate_config():
    """
    Validate the global and hyperparameter config files.
    """

    # Validate global config
    global_config = _get_config()
    for section in GLOBAL_CONFIG_SECTIONS:
        if section not in global_config.sections():
            raise ValueError(f"Section {section} not found in global config file '{GLOBAL_CONFIG_FILE_PATH}'")

    # Validate hyperparameter config
    hp_config = _get_config(get_current_hp_config_path())
    for section in HP_CONFIG_SECTIONS:
        if section not in hp_config.sections():
            raise ValueError(f"Section {section} not found in hyperparameter config file '{get_current_hp_config_path()}'")

    for key in Keys:
        config_sections = GLOBAL_CONFIG_SECTIONS if key.value < 1000 else HP_CONFIG_SECTIONS
        config_section_index = key.value // 100 if key.value < 1000 else (key.value - 1000) // 100
        file_path = GLOBAL_CONFIG_FILE_PATH if key.value < 1000 else get_current_hp_config_path()
        try:
            value = ""
            if key.value < 1000:
                value = global_config[config_sections[config_section_index]][key.name.lower()]
            else:
                value = hp_config[config_sections[config_section_index]][key.name.lower()]
            if value == "":
                raise ValueError(f"Key '{key.name.lower()}' under section '{config_sections[config_section_index]}' in '{file_path}' has no value in config file!")
        except KeyError:
            raise KeyError(f"Key '{key.name.lower()}' not found in config file under section '{config_sections[config_section_index]}' in '{file_path}'!")
        except configparser.NoOptionError:
            raise ValueError(f"Key '{key.name.lower()}' under section '{config_sections[config_section_index]}' in '{file_path}' has no value in config file!")


def config(key: Keys) -> Union[str, int, float, bool]:
    """
    This function is used to access global and hyperparameter configuration parameters that are used in the overall training and analysis process.

    Args:
        key (Keys): The key of the configuration parameter.

    Returns:
        Union[str, int, float, bool]: The value of the configuration parameter.
    """
    global __global_config_cache
    global __hp_config_cache

    # Check if the value is already cached
    if key.value < 1000:
        if key in __global_config_cache:
            return __global_config_cache[key]
    else:
        if key in __hp_config_cache:
            return __hp_config_cache[key]

    # Check if the key is a global config key or a hyperparameter config key
    config_path = GLOBAL_CONFIG_FILE_PATH if key.value < 1000 else get_current_hp_config_path()
    config_sections = GLOBAL_CONFIG_SECTIONS if key.value < 1000 else HP_CONFIG_SECTIONS
    config_section_index = key.value // 100 if key.value < 1000 else (key.value - 1000) // 100

    value = ""
    try:
        value = _get_config(config_path)[config_sections[config_section_index]][key.name.lower()].strip()
    except KeyError:
        raise KeyError(f"Key {key.name.lower()} not found in config file under section {config_sections[config_section_index]} in '{config_path}'")
    except configparser.NoOptionError:
        raise ValueError(f"Key {key.name.lower()} has no value in config file")
    except Exception as e:
        raise e

    # Try to parse to float, int etc.
    if value.find(".") != -1:
        try:
            value = float(value)
        except ValueError:
            pass
    else:
        try:
            value = int(value)
        except ValueError:
            pass

    if value == "True":
        value = True
    elif value == "False":
        value = False

    # Cache the value
    if key.value < 1000:
        __global_config_cache[key] = value
    else:
        __hp_config_cache[key] = value

    return value


def print_config():
    logging.debug(f"Global config ({GLOBAL_CONFIG_FILE_PATH}):")
    config_ = _get_config()
    for section in config_.sections():
        logging.debug(f"[{section}]")
        for name, value in config_[section].items():
            key = Keys[name.upper()]
            datatype = type(config(key)).__name__
            logging.debug(f"{name} = {value} <{datatype}>")
        logging.debug("")

    logging.debug(f"Hyperparameter config ({get_current_hp_config_path()}):")
    config_ = _get_config(get_current_hp_config_path())
    for section in config_.sections():
        logging.debug(f"[{section}]")
        for name, value in config_[section].items():
            key = Keys[name.upper()]
            datatype = type(config(key)).__name__
            logging.debug(f"{name} = {value} <{datatype}>")
        logging.debug("")


if __name__ == "__main__":
    """
    This script is used to create a new config file with all keys from the Keys enum.
    """

    CONFIG_SECTIONS = GLOBAL_CONFIG_SECTIONS
    CONFIG_FILE_PATH = GLOBAL_CONFIG_FILE_PATH

    config_ = configparser.ConfigParser()
    hp_config_ = configparser.ConfigParser()

    for section in GLOBAL_CONFIG_SECTIONS:
        config_.add_section(section)
    for section in HP_CONFIG_SECTIONS:
        hp_config_.add_section(section)
    for key in Keys:
        config_sections = GLOBAL_CONFIG_SECTIONS if key.value < 1000 else HP_CONFIG_SECTIONS
        config_section_index = key.value // 100 if key.value < 1000 else (key.value - 1000) // 100
        if key.value < 1000:
            config_[config_sections[config_section_index]][key.name.lower()] = ""
        else:
            hp_config_[config_sections[config_section_index]][key.name.lower()] = ""

    with open("config.empty.ini", "w") as configfile:
        config_.write(configfile)

    with open(os.path.join(HP_CONFIGS_PATH, "empty.ini"), "w") as configfile:
        hp_config_.write(configfile)

    logging.debug(f"Succesfully created config files.")
