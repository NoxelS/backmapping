import os
import sys
from ctypes import ArgumentError

from library.analysis.plots import plot_cluster_hist
from library.config import Keys, config

plot_names = [
    "",
    "training_loss.png",
    "training_mse.png",
    "training_lr.png",
    "training_mae.png",
    "training_val_acc.png",
    "training_val_loss.png",
    "training_val_mae.png",
]

for i in range(1, 7):
    plot_cluster_hist(i).savefig(os.path.join(os.path.join(config(Keys.DATA_PATH), "hist"), plot_names[i]), **{"dpi": 300, "bbox_inches": "tight"})
    print(f"Saved {plot_names[i]}")
