import os
import sys
from ctypes import ArgumentError

from library.analysis.plots import plot_cluster_hist
from library.config import Keys, config

# Get arguments
args = sys.argv[1:]

if len(args) != 1:
    raise ArgumentError("Please provide the index of the plot that should be created as an argument")

plot_names = [
    "training_loss.png",
    "training_lr.png",
    "training_mae.png",
    "training_val_acc.png",
    "training_val_loss.png",
    "training_val_mae.png",
]

plot_cluster_hist(args[0]).savefig(os.path.join(os.path.join(config(Keys.DATA_PATH), "hist"), plot_names[args[0]]), {"dpi": 300, "bbox_inches": "tight"})
