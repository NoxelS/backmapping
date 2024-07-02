import argparse
import logging
import os
import pickle
import socket
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, get_ic_type_from_index, ic_to_hlabel
from library.plot_config import set_plot_config

if not os.path.exists("post_analysis"):
    os.makedirs("post_analysis")


def post_analysis(config_name: str, type: str = "mixed", skip_single_plots: bool = False, custom_range: list = None):
    """
    Plot the histograms of the validation predictions for a specific configuration.

    Args:
        config_name (str): The name of the configuration to plot.
        type (str, optional): The type of the internal coordinate to plot. Defaults to "mixed". Can be "angle", "dihedral" or "bond".

    """

    # List all folders in analysis
    folders = os.listdir(os.path.join(config(Keys.DATA_PATH), "analysis"))
    folders = [f for f in folders if os.path.isdir(os.path.join(config(Keys.DATA_PATH), "analysis", f))]

    # Filter for name
    folders = [f for f in folders if config_name in f]

    # Get all the validation predictions
    val_pred_files = [f for f in [os.path.join(config(Keys.DATA_PATH), "analysis", f, "validation_predictions.pkl") for f in folders] if os.path.exists(f)]
    print("Found the following predictions:", "\n - ".join(["", *val_pred_files]))
    print([get_ic_type_from_index(int((os.path.basename(os.path.dirname(f)))[::-1].split("_", 1)[0][::-1])) for f in val_pred_files])
    # Filter out any files that do not match type type
    val_pred_files = [f for f in val_pred_files if get_ic_type_from_index(int((os.path.basename(os.path.dirname(f)))[::-1].split("_", 1)[0][::-1])) == type]

    print("Found the following predictions:", "\n - ".join(["", *val_pred_files]))

    # Load all the predictions into a dictionary
    predictions = {}
    for file in val_pred_files:
        with open(file, "rb") as f:
            predictions[os.path.basename(os.path.dirname(file))] = pickle.load(f, fix_imports=False)

    if not skip_single_plots:
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
            axs[1].set_ylim(-0.1, 0.1)

            plt.savefig(os.path.join("post_analysis", f"{model}_histogram.pdf"))

    # Plot all the histograms in one plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Make upper plot higher
    axs[0].set_position([0.1, 0.5, 0.8, 0.4])
    axs[1].set_position([0.1, 0.1, 0.8, 0.2])

    set_plot_config(["seaborn-pastel"])

    y_true = []
    y_pred = []

    for model in predictions.keys():
        ic_index = int(model[::-1].split("_", 1)[0][::-1])
        ic_label = ic_to_hlabel(get_ic_from_index(ic_index))

        y_true_ = np.array(predictions[model][:, 1])
        y_pred_ = np.array(predictions[model][:, 2])

        try:
            y_true_ = [y[0] for y in y_true_]
            y_pred_ = [y[0] for y in y_pred_]
        except Exception as _:
            pass

        y_true.extend(y_true_)
        y_pred.extend(y_pred_)

    if custom_range is not None:
        y_true = [y for y in y_true if y >= custom_range[0] and y <= custom_range[1]]
        y_pred = [y for y in y_pred if y >= custom_range[0] and y <= custom_range[1]]

    # Normalize the values
    y_true_weights = np.ones_like(y_true) / float(len(y_true))
    y_pred_weights = np.ones_like(y_pred) / float(len(y_pred))

    min_y = min(min(y_true), min(y_pred)) if custom_range is None else custom_range[0]
    max_y = max(max(y_true), max(y_pred)) if custom_range is None else custom_range[1]

    # Make two plots, the upper plot shows the two histograms while
    # the lower plot shows the difference between the two histograms
    a, bins, _ = axs[0].hist(y_true, bins=250, alpha=0.4, label="True", color="xkcd:blue", edgecolor="black", weights=y_true_weights)
    b, _, _ = axs[0].hist(y_pred, bins=bins, alpha=0.4, label="Predicted", color="xkcd:green", edgecolor="black", weights=y_pred_weights)

    axs[0].legend()

    # Set xlabel
    x_label = "IC Value (a.u.)"
    if type == "angle" or type == "dihedral":
        x_label = "Angle (°)"
    elif type == "bond":
        x_label = "Bond Length (Å)"

    # Add labels
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Rel. Frequency")
    axs[0].grid(which="major", linestyle="-", linewidth="0.75", color=[0.1, 0.1, 0.1], alpha=0.5)

    # Lower plot
    axs[1].bar(bins[:-1], height=((a) - (b)), alpha=0.4, color="xkcd:purple", edgecolor="black", width=bins[1] - bins[0])
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("Rel. Frequency Diff.")

    # Set ranges
    axs[0].set_xlim(min_y, max_y)
    axs[1].set_xlim(min_y, max_y)
    axs[1].set_ylim(-0.1, 0.1)

    # Title
    title = "Histogram of All ICs"
    if type == "angle":
        title = "Histogram of All Angles"
    elif type == "dihedral":
        title = "Histogram of All Dihedral Angles"
    elif type == "bond":
        title = "Histogram of All Bond Lengths"

    axs[0].set_title(title)

    plt.savefig(os.path.join("post_analysis", f"all_histograms_{config_name}_{type}{f'_cr_{custom_range[0]}_{custom_range[1]}' if custom_range is not None else ''}.pdf"))
    plt.close()


post_analysis("prod", "dihedral", skip_single_plots=True, custom_range=[90, 100])
post_analysis("prod", "dihedral", skip_single_plots=True, custom_range=[72, 82])
post_analysis("prod", "dihedral", skip_single_plots=True, custom_range=[110, 120])
post_analysis("prod", "dihedral", skip_single_plots=True)

post_analysis("prod", "angle", skip_single_plots=True)
post_analysis("prod", "angle", skip_single_plots=True, custom_range=[120, 130])
post_analysis("prod", "angle", skip_single_plots=True, custom_range=[110, 115])

post_analysis("prod", "bond", skip_single_plots=True)
post_analysis("prod", "bond", skip_single_plots=True, custom_range=[1.475, 1.515])
post_analysis("prod", "bond", skip_single_plots=True, custom_range=[1.515, 1.575])
post_analysis("prod", "bond", skip_single_plots=True, custom_range=[1.575, 1.62])
