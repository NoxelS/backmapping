import csv
import os
import sys
import time

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
