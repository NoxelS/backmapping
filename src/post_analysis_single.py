import logging
import os
import pickle
import socket
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, ic_to_hlabel
from library.plot_config import set_plot_config

if not os.path.exists("post_analysis"):
    os.makedirs("post_analysis")

# List all folders in analysis
folders = os.listdir(os.path.join(config(Keys.DATA_PATH), "analysis"))
folders = [f for f in folders if os.path.isdir(os.path.join(config(Keys.DATA_PATH), "analysis", f))]

# Filter for name
folders = [f for f in folders if "prod" in f]

# Get all the validation predictions
val_pred_files = [f for f in [os.path.join(config(Keys.DATA_PATH), "analysis", f, "validation_predictions.pkl") for f in folders] if os.path.exists(f)]

print("Found the following predictions:", "\n - ".join(["", *val_pred_files]))

# Load all the predictions into a dictionary
predictions = {}
for file in val_pred_files:
    with open(file, "rb") as f:
        predictions[os.path.basename(os.path.dirname(file))] = pickle.load(f, fix_imports=False)


for model in predictions.keys():
    print("Model:", model)

    ic_index = int(model[::-1].split("_", 1)[0][::-1])
    ic_label = ic_to_hlabel(get_ic_from_index(ic_index))

    y_true = np.array(predictions[model][:, 1])
    y_pred = np.array(predictions[model][:, 2])

    try:
        y_true = [y[0] for y in y_true]
        y_pred = [y[0] for y in y_pred]
    except Exception as _:
        pass

    # Normalize the values
    y_true_weights = np.ones_like(y_true) / float(len(y_true))
    y_pred_weights = np.ones_like(y_pred) / float(len(y_pred))

    min_y = min(min(y_true), min(y_pred))
    max_y = max(max(y_true), max(y_pred))

    # Make two plots, the upper plot shows the two histograms while
    # the lower plot shows the difference between the two histograms
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Make upper plot higher
    axs[0].set_position([0.1, 0.5, 0.8, 0.4])
    axs[1].set_position([0.1, 0.1, 0.8, 0.2])

    set_plot_config(["seaborn-pastel"])

    a, bins, _ = axs[0].hist(y_true, bins=50, alpha=0.4, label="True", color="xkcd:blue", edgecolor="black", weights=y_true_weights)
    b, _, _ = axs[0].hist(y_pred, bins=bins, alpha=0.4, label="Predicted", color="xkcd:green", edgecolor="black", weights=y_pred_weights)
    axs[0].legend()

    # Add labels
    axs[0].set_xlabel("Angle (°)" if "angle" in model else "Bond Length (Å)")
    axs[0].set_ylabel("Rel. Frequency")
    axs[0].grid(which="major", linestyle="-", linewidth="0.75", color=[0.1, 0.1, 0.1], alpha=0.5)
    axs[0].grid(which="minor", linestyle="--", linewidth="0.25", color=[0.2, 0.2, 0.2], alpha=0.3)
    title_a = "Bond Length" if "bond" in model else "Angle"
    title = title_a + " Histogram for IC " + ic_label
    axs[0].set_title(title)

    # Lower plot
    axs[1].bar(bins[:-1], height=((a) - (b)), alpha=0.4, color="xkcd:purple", edgecolor="black", width=bins[1] - bins[0])
    axs[1].set_xlabel("Angle (°)" if "angle" in model else "Bond Length (Å)")
    axs[1].set_ylabel("Rel. Frequency Diff.")

    # Set ranges
    axs[0].set_xlim(min_y, max_y)
    axs[1].set_xlim(min_y, max_y)

    plt.savefig(os.path.join("post_analysis", f"{model}_histogram.pdf"))
