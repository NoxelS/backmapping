import configparser
import os
from enum import Enum

CONFIG_FILE_PATH = "config.ini"
CONFIG_SECTIONS = [_.upper() for _ in ["global", "train_master", "email", "training"]]  # Every section can hold up to 100 keys


class Keys(Enum):
    # Global config
    DATA_PATH = 0

    # Master resolver config
    PORT = 100

    # Email config
    EMAIL_SERVER = 200
    EMAIL_TARGET = 201
    EMAIL_USER = 202

    # Training config
    BATCH_SIZE = 300
    VALIDATION_SPLIT = 301
    NEIGHBORHOOD_SIZE = 302
    EPOCHS = 303
    MODEL_NAME_PREFIX = 304
    DATA_USAGE = 305
    USE_TENSORBOARD = 306


def _removeInlineComments(cfgparser):
    for section in cfgparser.sections():
        for item in cfgparser.items(section):
            cfgparser.set(section, item[0], item[1].split(";")[0].strip())


def _get_config() -> configparser.ConfigParser:
    config_ = configparser.ConfigParser()
    config_.read(filenames="config.ini", encoding="utf-8")
    _removeInlineComments(config_)
    return config_


def config(key: Keys) -> float | int | bool | str | None:
    value = ""
    try:
        value = _get_config()[CONFIG_SECTIONS[key.value // 100]][key.name.lower()].strip()
    except KeyError:
        # Retuzrn null
        value = None

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

    return value


if __name__ == "__main__":
    config_ = configparser.ConfigParser()

    for section in CONFIG_SECTIONS:
        config_.add_section(section)
    for key in Keys:
        config_[CONFIG_SECTIONS[key.value // 100]][key.name.lower()] = ""

    # Prompt user if config file already exists
    if os.path.exists(CONFIG_FILE_PATH):
        print(f"Config file already exists at {CONFIG_FILE_PATH}. Do you want to overwrite it? (y/n)")
        if input().lower() != "y":
            exit()

    with open(CONFIG_FILE_PATH, "w") as configfile:
        config_.write(configfile)

    print(f"Succesfully created config file at {CONFIG_FILE_PATH} with {len(Keys)} keys")
