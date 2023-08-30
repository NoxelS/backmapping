import os
import numpy as np
import matplotlib.pyplot as plt

# Plot the training history of all models in a single plot

PATH_TO_HIST = os.path.join("data", "hist")

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

# Make full screen
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Loop over all files in the hist folder
for i, hist in enumerate(os.listdir(PATH_TO_HIST)):
    # file is named: training_history_C1.csv
    atom_name = hist.split("_")[2].split(".")[0]
    mean_distance = mean_distances[atom_name]
    # Load csv
    hist = np.loadtxt(os.path.join(PATH_TO_HIST, hist), delimiter=",", skiprows=1)
    # Plot
    ax.plot(hist[:, 0], hist[:, 2], label=atom_name, color=plt.cm.cool(mean_distance))
    
# Add labels
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training History")

# Plot legend outside of plot in two columns
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)

# Add legend that explains color to the right of the plot
ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
cmap = plt.cm.cool
norm = plt.Normalize(vmin=0, vmax=1)
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax2, orientation='vertical')
ax2.set_title("NMD")


plt.tight_layout()

# Save
plt.savefig("training_history.png")