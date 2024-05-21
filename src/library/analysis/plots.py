import logging
import os
import pickle
import socket
import sys
import time
from operator import inv
from re import T
from tkinter import font
from turtle import color

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
from pyparsing import line
from xarray import align

from library.classes.generators import inverse_scale_output_ic
from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, get_ic_type_from_index, ic_to_hlabel
from library.plot_config import set_plot_config

PATH_TO_HIST = os.path.join(config(Keys.DATA_PATH), "hist")


def plot_cluster_hist(data_col=2):

    # Load plot config
    set_plot_config()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    min_loss = 99999

    hist_files = os.listdir(PATH_TO_HIST)

    # Data to find the average graph
    avg_data = []

    # Loop over all files in the hist folder
    for i, hist in enumerate(hist_files):
        # file is named: training_history_prefix_ic.csv
        ic_index = hist.split("_")[-1].split(".")[0]

        try:
            # Load csv
            hist = np.loadtxt(os.path.join(PATH_TO_HIST, hist), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        except Exception as e:
            continue

        # If only one row, add dimension
        if len(hist.shape) == 1:
            hist = hist.reshape(1, -1)

        # There are maybe multiple train cylces so reindex the epochs accordingly
        hist[:, 0] = np.arange(hist.shape[0])

        if np.min(hist[:, int(data_col)]) < min_loss:
            min_loss = np.min(hist[:, int(data_col)])

        # Plot
        # color = plt.cm.cool(mean_distance)
        ax.plot(hist[:, 0] + 1, hist[:, int(data_col)], label=str(ic_index))

        # Add to average data
        avg_data.append(hist[:, int(data_col)])

    # Plot average avg_data = list for every atom with the history of the loss
    max_epoch = np.max([len(i) for i in avg_data])
    avg_data_y = [[] for _ in range(max_epoch)]
    for atom_data in avg_data:
        for epoch, data in enumerate(atom_data):
            avg_data_y[epoch].append(data)

    avg_data = np.array([np.mean(i) for i in avg_data_y])

    # Plot average
    # ax.plot(np.arange(len(avg_data)) + 1, avg_data, label="Average", color="black", alpha=0.5)

    # Get name of data
    data_name = ["", "Accuracy", "MSE Loss (Å)", "Learning Rate", "Mean Average Error", "Val. Accuracy", "Val. MSE Loss (Å)", "Val. Mean Average Error"][int(data_col)]

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(data_name)
    ax.set_title("Training History")

    # # Add grid
    # ax.grid(axis="y", alpha=0.5)
    # ax.grid(axis="x", alpha=0.5)

    # Make log scale
    ax.set_yscale("log")

    # Add minor tick labels
    ax.minorticks_on()

    # Add 10 y-ticks between min and max
    # ax.set_yticks(np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10))

    # # Add 10 y-tick labels
    # ax.set_yticklabels([f"{i:.2f}" for i in np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10)])

    # Add line where the minimum loss is
    # ax.axhline(y=min_loss, color="black", linestyle="--", alpha=0.4)

    # Add label
    # text_center_y = min_loss - 0.66 * (min_loss - ax.get_ylim()[0])
    # text_center_x = ax.get_xlim()[1] / 2
    # ax.text(
    #     text_center_x, text_center_y, f"Minimum Loss: {min_loss:.2f} Å", horizontalalignment="center", verticalalignment="bottom", fontsize=12, color="black", alpha=0.75
    # )

    # Plot legend outside of plot in two columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=2)
    ax.get_legend().set_title("Atom Names")

    # Add legend that explains color to the bottom
    # ax2 = fig.add_axes([0.93, 0.11, 0.2, 0.05])
    # cmap = plt.cm.cool
    # norm = plt.Normalize(vmin=0, vmax=1)
    # cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation="horizontal")
    # ax2.set_title("NMD")  # Normalized Mean Distance

    return fig


def plot_hist_single(ic_index: int):
    # Load plot config
    set_plot_config()
    fig, ax = plt.subplots()

    ic_type = get_ic_type_from_index(ic_index)
    dim = "Å" if ic_type == "bond" else "°"

    # Find the file
    target_csv_file = [file for file in os.listdir(PATH_TO_HIST) if f"{ic_index}.csv" in file][0]

    # Load csv
    hist = np.loadtxt(os.path.join(PATH_TO_HIST, target_csv_file), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

    # If only one row, add dimension
    if len(hist.shape) == 1:
        hist = hist.reshape(1, -1)

    hist_epoch, hist_loss, hist_lr, hist_mae, hist_mse, hist_val_loss, hist_val_mae, hist_val_mse = (
        hist[:, 0],
        hist[:, 1],
        hist[:, 2],
        hist[:, 3],
        hist[:, 4],
        hist[:, 5],
        hist[:, 6],
        hist[:, 7],
    )

    # Scale the losses accordingly
    hist_loss = inverse_scale_output_ic(ic_index, hist_loss) - inverse_scale_output_ic(ic_index, 0)
    hist_val_loss = inverse_scale_output_ic(ic_index, hist_val_loss) - inverse_scale_output_ic(ic_index, 0)

    # There are maybe multiple train cylces so reindex the epochs accordingly
    hist_epoch = np.arange(hist.shape[0])

    # Calculate the minimum loss
    min_loss = np.min(hist_loss)
    min_val_loss = np.min(hist_val_loss)

    # Lr_pices defines the epochs where the lr changed
    lr_pices = [i for i in range(1, len(hist_lr)) if hist_lr[i] != hist_lr[i - 1]]
    lr_pices.insert(0, 0)
    lr_pices.append(len(hist_lr) - 1)

    # Make colormap for the lr
    cmap = plt.cm.plasma
    # Normalize the lr
    norm = plt.Normalize(vmin=min(hist_lr), vmax=max(hist_lr))
    # Make colors for each epoch
    colors = [cmap(norm(i)) for i in hist_lr]

    max_y = max(max(hist_loss), max(hist_val_loss))
    min_y = min(min(hist_loss), min(hist_val_loss))

    # Add h-line for each lr change
    for i in lr_pices[1:-1]:
        ax.axvline(x=i, color=colors[i], linestyle=":", alpha=0.4, linewidth=1.25)
        ax.text(
            i - 0.1,
            max_y - 0.2 * (max_y - min_y),
            f"lr: {hist_lr[i]:.1e}",
            rotation=90,
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=12,
            color="black",
            fontweight=500,
            alpha=0.75,
        )

    # Plot minimum loss
    ax.axhline(y=min_loss, color="black", linestyle="-", alpha=0.5, linewidth=0.75)
    ax.text(
        1,
        min_loss,
        f"Min. Loss: {min_loss:.4f}{dim}",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        color="black",
        fontweight=500,
        alpha=0.75,
    )

    # Plot minimum loss
    ax.axhline(y=min_val_loss, color="black", linestyle="--", alpha=0.5, linewidth=0.75)
    ax.text(
        1,
        min_val_loss,
        f"Min. Val. Loss: {min_val_loss:.4f}{dim}",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        color="black",
        fontweight=500,
        alpha=0.75,
    )

    # Plot the loss and val loss picewise for the lr
    for i in range(len(lr_pices) - 1):
        interval = (lr_pices[i], lr_pices[i + 1] + 1)
        lr = hist_lr[interval[0]]

        # Plot the loss
        ax.plot(
            hist_epoch[interval[0] : interval[1]],
            hist_loss[interval[0] : interval[1]],
            color=colors[lr_pices[i]],
            # marker="s",
            markeredgecolor="black",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1,
        )
        ax.plot(
            hist_epoch[interval[0] : interval[1]],
            hist_val_loss[interval[0] : interval[1]],
            color=colors[lr_pices[i]],
            # marker="v",
            markeredgecolor="black",
            markerfacecolor="white",
            linestyle="--",
            linewidth=1,
        )

    # Make pseudo plot for the legend
    ax.plot(
        [],
        [],
        color="black",
        marker="s",
        linestyle="-",
        label="Training Loss",
        markerfacecolor="white",
    )
    ax.plot(
        [],
        [],
        color="black",
        marker="v",
        linestyle="--",
        label="Validation Loss",
        markerfacecolor="white",
    )

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({dim})")

    # Plot legend outside of plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Plot colorscale for the lr on the right
    ax2 = fig.add_axes([0.94, 0.11, 0.02, 0.57])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation="vertical")
    ax2.set_title("Learning Rate", fontsize=12, color="black", fontweight=600, alpha=0.75, ha="left")
    # ax2.set_yscale("log")

    # Set the y-ticks for the colorbar
    # ax2.set_yticks(hist_lr[lr_pices])
    ax2.get_yaxis().get_major_formatter().labelOnlyBase = False

    ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.set_yscale("log")

    # ax.set_xticks(np.arange(len(hist_epoch)))

    title = f"Training History for {ic_type.capitalize()} {'Angle ' if ic_type == 'dihedral' else ''}{ic_to_hlabel(get_ic_from_index(ic_index))}"
    ax.set_title(title)

    fig.savefig("test.pdf")
