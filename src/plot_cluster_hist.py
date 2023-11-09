import os

import numpy as np
import matplotlib.pyplot as plt

from library.config import Keys, config
from library.classes.generators import ABSOLUT_POSITION_SCALE

PATH_TO_HIST = os.path.join(config(Keys.DATA_PATH), "hist")

# Plot the training history of all models in a single plot
def plot_cluster_hist():
    # Get mean distances to color the plot accordingly
    mean_distances = {}

    # Read csv
    with open(os.path.join(config(Keys.DATA_PATH), "mean_distances.csv"), "r") as f:
        # Skip header
        f.readline()

        for line in f.readlines():
            # Split line
            line = line.split(",")
            # Get mean distance
            mean_distances[line[0]] = float(line[1])

    # Normalize mean distances
    mean_distances = {k: v / max(mean_distances.values()) for k, v in mean_distances.items()}

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    min_loss = 99999

    # Loop over all files in the hist folder
    for i, hist in enumerate(os.listdir(PATH_TO_HIST)):
        # file is named: training_history_C1.csv
        atom_name = hist.split("_")[2].split(".")[0]
        mean_distance = mean_distances[atom_name]

        try:
            # Load csv
            hist = np.loadtxt(os.path.join(PATH_TO_HIST, hist), delimiter=",", skiprows=1, usecols=(0, 1, 2))
        except Exception as e:
            continue

        # If only one row, add dimension
        if len(hist.shape) == 1:
            hist = hist.reshape(1, -1)

        # There are maybe multiple train cylces so reindex the epochs accordingly
        hist[:, 0] = np.arange(hist.shape[0])


        if np.min(hist[:, 2] * ABSOLUT_POSITION_SCALE) < min_loss:
            min_loss = np.min(hist[:, 2] * ABSOLUT_POSITION_SCALE)

        # Plot
        ax.plot(hist[:, 0] + 1, hist[:, 2] * ABSOLUT_POSITION_SCALE, label=atom_name, color=plt.cm.cool(mean_distance))

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (Å)")
    ax.set_title("Training History")

    # Make log scale
    ax.set_yscale("log")
    
    # Add 10 y-ticks between min and max
    ax.set_yticks(np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10))
    
    # Add 10 y-tick labels
    ax.set_yticklabels([f"{i:.2f}" for i in np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 10)])
    
    # Add line where the minimum loss is
    ax.axhline(y=min_loss, color="black", linestyle="--", alpha=0.4)
    
    # Add label
    ax.text(2, min_loss, f"Minimum Loss: {min_loss:.2f} Å", horizontalalignment='center', verticalalignment='bottom', fontsize=12, color="black", alpha=0.4)

    # Plot legend outside of plot in two columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    ax.get_legend().set_title("Atom Names")

    # Add legend that explains color to the bottom
    ax2 = fig.add_axes([0.93, 0.11, 0.2, 0.05])
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation='horizontal')
    ax2.set_title("NMD") # Normalized Mean Distance

    # Save
    plt.savefig("training_history.png",bbox_inches='tight')


if __name__ == "__main__":
    plot_cluster_hist()