import sys
import time
import os

# Quick and dirty way to color the atoms by their element
DEFAULT_ELEMENT_COLOR_MAP = {
    'H':    [1, 1, 1],
    'C':    [0.0, 0.0, 0.0],
    'N':    [0.0, 0.0, 1],
    'O':    [1, 0.0, 0.0],
    'P':    [1, 0.5, 0.0],
    'S':    [1.0, 1.0, 0.0],
    'NA':   [0.9, 0.5, 0.9],
    'CL':   [0.0, 1, 0.0],
    'MG':   [0.0, 0.0, 0.0],
    'CA':   [0.0, 0.0, 0.0],
    'FE':   [0.0, 0.0, 0.0],
    'ZN':   [0.0, 0.0, 0.0],
    'CU':   [0.0, 0.0, 0.0],
    'MN':   [0.0, 0.0, 0.0],
    'K':    [0.9, 0.5, 0.9],
    'F':    [0.0, 1, 0.0],
    'BR':   [0.0, 0.0, 0.0],
    'I':    [0.0, 0.0, 0.0],
    'CD':   [0.0, 0.0, 0.0],
    'CO':   [0.0, 0.0, 0.0]
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