from library.classes.generators import ABSOLUT_POSITION_SCALE
import matplotlib.pyplot as plt
import numpy as np
import os

PATH_TO_HIST = os.path.join("data", "hist")

# Plot the training history of all models in a single plot
def plot_cluster_hist():
    # Get mean distances to color the plot accordingly
    mean_distances = {}

    # Read csv
    with open(os.path.join("data", "mean_distances.csv"), "r") as f:
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

    # Loop over all files in the hist folder
    for i, hist in enumerate(os.listdir(PATH_TO_HIST)):
        # file is named: training_history_C1.csv
        atom_name = hist.split("_")[2].split(".")[0]
        mean_distance = mean_distances[atom_name]

        try:
            # Load csv
            hist = np.loadtxt(os.path.join(PATH_TO_HIST, hist), delimiter=",", skiprows=1)
        except:
            print(f"Could not load {hist}, skipping...")
            continue

        # If only one row, add dimension
        if len(hist.shape) == 1:
            hist = hist.reshape(1, -1)

        # There are maybe multiple train cylces so only start plotting after finding the last 0 epoch
        for i, row in enumerate(hist):
            if int(row[0]) == 0:
                hist = hist[i:]

        # Plot
        ax.plot(hist[:, 0] + 1, hist[:, 2] * ABSOLUT_POSITION_SCALE, label=atom_name, color=plt.cm.cool(mean_distance))

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (Å)")
    ax.set_title("Training History")

    # Make log scale
    ax.set_yscale("log")

    # Plot legend outside of plot in two columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    ax.get_legend().set_title("Atom Names")

    # Add legend that explains color to the bottom
    ax2 = fig.add_axes([0.78, 0.07, 0.2, 0.05])
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation='horizontal')
    ax2.set_title("NMD") # Normalized Mean Distance

    # Save
    plt.savefig("training_history.png",bbox_inches='tight')


if __name__ == "__main__":
    plot_cluster_hist()