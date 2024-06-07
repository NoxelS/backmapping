import csv
import os
import sys
import time
from typing import Union
import numpy as np

from library.datagen.topology import get_ic_from_index, get_ic_type

# Quick and dirty way to color the atoms by their element
DEFAULT_ELEMENT_COLOR_MAP = {
    "H": [1, 1, 1],
    "C": [0.0, 0.0, 0.0],
    "N": [0.0, 0.0, 1],
    "O": [1, 0.0, 0.0],
    "P": [1, 0.5, 0.0],
    "S": [1.0, 1.0, 0.0],
    "NA": [0.9, 0.5, 0.9],
    "CL": [0.0, 1, 0.0],
    "MG": [0.0, 0.0, 0.0],
    "CA": [0.0, 0.0, 0.0],
    "FE": [0.0, 0.0, 0.0],
    "ZN": [0.0, 0.0, 0.0],
    "CU": [0.0, 0.0, 0.0],
    "MN": [0.0, 0.0, 0.0],
    "K": [0.9, 0.5, 0.9],
    "F": [0.0, 1, 0.0],
    "BR": [0.0, 0.0, 0.0],
    "I": [0.0, 0.0, 0.0],
    "CD": [0.0, 0.0, 0.0],
    "CO": [0.0, 0.0, 0.0],
    # ...
}


def log(*args, **kwargs):
    ts = time.strftime("[%H:%M:%S]:", time.localtime())
    print(ts, *args, **kwargs)


def log_progress(func_name):
    """
    Logs the progress of a function by printing the time it took to execute it.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):

            t0 = time.time()
            log(f"Starting {func_name}...")
            result = function(*args, **kwargs)
            log(f"Finished {func_name} after {time.time() - t0:.2f}s")

            return result

        return wrapper

    return decorator


# Print iterations progress
def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=50, fill="#", printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "~" * (length - filledLength)

    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def print_matrix(matrix):
    """
    This function prints a matrix in ascii art.

    Args:
        matrix (list): List with shape (batch_size, i, j, 1)
    """
    # Prints an output/input matrix in ascii art
    for i in range(matrix.shape[0]):
        print(" ", end="")
        for k in range(matrix.shape[1]):
            print(f"{k:6d}", end="  ")
        print()
        print(matrix.shape[1] * 8 * "-")
        # Batch
        for j in range(matrix.shape[2]):
            # Y
            for k in range(matrix.shape[1]):
                # X
                minus_sign_padding = " " if matrix[i, k, j, 0] >= 0 else ""
                print(f"{minus_sign_padding}{matrix[i, k, j, 0]:.4f}", end=" ")
            print()
        print(matrix.shape[1] * 8 * "-")


def print_input_matrix(matrix, padding=0):
    """
    This function prints a matrix in ascii art.

    Args:
        matrix (list): List with shape (batch_size, i, j, 1)
    """
    # Prints an output/input matrix in ascii art
    for i in range(matrix.shape[0]):
        print(" ", end="")
        for k in range(matrix.shape[1]):
            print(f"{k:6d}", end="  ")
        print()
        print((matrix.shape[1] * 8 + 8) * "-")
        # Batch
        for j in range(matrix.shape[2]):
            if j >= padding and j < matrix.shape[2] - padding:
                print(f"[{['x','y','z'][(j-padding)%3]}{(j-padding)//3}]", end="")
            else:
                print(f"[  ]", end="")
            # Y
            for k in range(matrix.shape[1]):
                # X
                minus_sign_padding = " " if matrix[i, k, j, 0] >= 0 else ""
                print(f"{minus_sign_padding}{matrix[i, k, j, 0]:.4f}", end=" ")
                if (k + 1) == padding or (k + 1) == matrix.shape[1] - padding:
                    print(2 * " ", end="")
            print("")
            if (j + 1) == padding or (j + 1) == matrix.shape[2] - padding or (j - padding) % 3 == 2:
                print("")
        print((matrix.shape[1] * 8 + 8) * "-")


def to_significant(value, significant_digits=3):
    """
    This function converts a number to a string with a certain number of significant digits.

    Args:
        value (float): The number to convert.
        significant_digits (int): The number of significant digits to keep.

    Returns:
        str: The number as a string with the specified number of significant digits.
    """
    return "{:.{}g}".format(value, significant_digits)


def scale_output_ic(ic_index: int, value: float) -> Union[float, tuple]:
    """
    Scales the output internal coordinate value based on the internal coordinate index.
    """
    ic = get_ic_from_index(ic_index)
    ic_type = get_ic_type(ic)
    mean, std = float(ic["mean"]), float(ic["std"])

    if ic_type == "bond":
        # Bonds are between [1A, 1.7A] -> [0,1] -> [-0.25, 0.25]
        # return (value - 1) / (2 * (1.7 - 1)) - 0.25
        # Normalization
        return ((value - mean) / std) / 100

    elif ic_type == "angle":
        # Angles are between [90°, 160°] -> [0,1]
        # return (value - np.deg2rad(90)) / (np.deg2rad(160) - np.deg2rad(90))
        return [np.cos(value), np.sin(value)]
    elif ic_type == "dihedral":
        # Dihedrals are between [50°, 140°] -> [0,1]
        # return (value - np.deg2rad(50)) / (np.deg2rad(140) - np.deg2rad(50))
        # return (value - np.deg2rad(90)) / (np.deg2rad(160) - np.deg2rad(90))
        return [np.cos(value), np.sin(value)]

    else:
        raise Exception(f"Internal coordinate type {ic_type} not supported!")


def inverse_scale_output_ic(ic_index: int, value: Union[float, tuple]) -> float:
    """
    Inversely scales the output internal coordinate value based on the internal coordinate index.
    """
    ic = get_ic_from_index(ic_index)
    ic_type = get_ic_type(ic)
    mean, std = float(ic["mean"]), float(ic["std"])

    if ic_type == "bond":
        # Bonds are between [-0.25, 0.25] -> [0,1] -> [1A, 1.7A]
        # return (value + 0.25) * 2 * (1.7 - 1) + 1
        # Inverse normalization
        return 100 * value * std + mean
    elif ic_type == "angle":
        # Angles are [cos(phi), sin(phi)]
        return np.arctan2(value[1], value[0])
    elif ic_type == "dihedral":
        # Angles are [cos(phi), sin(phi)]
        return np.arctan2(value[1], value[0])
    else:
        raise Exception(f"Internal coordinate type {ic_type} not supported!")
