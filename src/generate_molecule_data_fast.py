from Bio.PDB.PDBIO import PDBIO
from datetime import datetime
from library.parser import get_cg_at_datasets

import os
import time
import numpy as np
import matplotlib.pyplot as plt

input_dir_path = "data/membranes"
output_dir_path = "data/molecules"
output_box_table_path = "data"    # <- This is the path where the box sizes are saved
training_dir_path = "data/training"

TRAINING_DATA_MODE = True
MAX_SAMPLES = 15 * (10 + 1) * 1024  # 102400
NEIGHBOUR_CUTOFF_RADIUS = 10.0     # The radius in which the neighbours are considered (in Angstrom)
TARGET_NEIGHBOUR_COUNT = 4        # The target number of neighbours for each residue, only used for logging

cg_datasets, at_datasets = [], []
io = PDBIO()
idx_at = 0
idx_cg = 0
current_time = time.time()


def elapsed_time():
    elapsed_time = time.time() - current_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


# Get all CG and AT datasets
cg_datasets, at_datasets = get_cg_at_datasets(
    input_dir_path, CG_PATTERN='cg.pdb', AT_PATTERN='at.pdb')

# Create output dir if it does not exist
if not os.path.exists(output_dir_path) and not TRAINING_DATA_MODE:
    os.makedirs(output_dir_path)

if TRAINING_DATA_MODE and not os.path.exists(training_dir_path):
    os.makedirs(training_dir_path)
    os.makedirs(f"{training_dir_path}/input")
    os.makedirs(f"{training_dir_path}/output")

# List to store all box sizes
cg_box_sizes = []
at_box_sizes = []

# List of tuples to keep track of which dataset has which box size
# Consists of (dataset_idx, box_size)
cg_box_size_dataset_relations = []
at_box_size_dataset_relations = []

# For finding neighbours
neighbours_counts = []
neighbourhoods = []     # This is a list of lists, where each list contains the neighbours of one residue. The index of the list is the residue index

# Loop over both at the same time (these are generators, so they are not loaded into memory immediately)
for i, (cg_dataset, at_dataset) in enumerate(zip(cg_datasets, at_datasets)):
    # Open the CG and AT files
    cg_file = open(cg_dataset.path)
    at_file = open(at_dataset.path)

    # Read the CG and AT files
    cg_lines = cg_file.readlines()
    at_lines = at_file.readlines()

    # Close the CG and AT files
    cg_file.close()
    at_file.close()

    # Find the crystal box size
    cg_box_size = [(float(line.split()[1]), float(line.split()[2]), float(line.split()[3])) for line in cg_lines if line.startswith("CRYST1")][0]
    at_box_size = [(float(line.split()[1]), float(line.split()[2]), float(line.split()[3])) for line in at_lines if line.startswith("CRYST1")][0]

    # Add the box size to the list
    cg_box_sizes.append(cg_box_size)
    at_box_sizes.append(at_box_size)

    # Print mean nad std of box sizes
    cg_box_mean = np.mean(cg_box_sizes, axis=0)
    cg_box_std = np.std(cg_box_sizes, axis=0)
    at_box_mean = np.mean(at_box_sizes, axis=0)
    at_box_std = np.std(at_box_sizes, axis=0)

    # Remove all lines that are not ATOM lines
    cg_lines = [line for line in cg_lines if line.startswith("ATOM")]
    at_lines = [line for line in at_lines if line.startswith("ATOM")]

    last_residue_idx_cg = 0
    last_residue_idx_at = 0

    line_stack_cg = []
    line_stack_at = []

    coordinates_cg = []  # Only used to calculate COM
    cg_com_list = []    # Used to store all COMs of one membrane. Each entry is a tuple of (COM, idx)

    # Loop over the lines of the CG files
    for j, cg_line in enumerate(cg_lines):
        # Check if we have reached the max number of samples
        if idx_cg > MAX_SAMPLES:
            break

        # Split the lines and get the residue index
        cg_line_split = cg_line.split()
        residue_idx_cg = int(cg_line_split[4])

        # Get coordinates of all atoms in the residue
        coordinates_cg.append((float(cg_line_split[5]), float(cg_line_split[6]), float(cg_line_split[7])))

        # Check if the residue changes or last line
        if residue_idx_cg != last_residue_idx_cg or j == len(cg_lines) - 1:
            # Add the line to the stack
            if j == len(cg_lines) - 1:
                line_stack_cg.append(cg_line)

            # Make folder for the residue if not in training data mode
            if not TRAINING_DATA_MODE:
                if not os.path.exists(f"{output_dir_path}/{i}_{last_residue_idx_cg}"):
                    os.makedirs(f"{output_dir_path}/{i}_{last_residue_idx_cg}")

            # Add TER line to the stack
            line_stack_cg.append(f"TER      13      DOPC    {last_residue_idx_cg}\n")  # <- I don't know why this is needed, but biopython does it too
            line_stack_cg.append("END   ")

            # Set path
            save_path = f"{output_dir_path}/{i}_{last_residue_idx_cg}/cg.pdb" if not TRAINING_DATA_MODE else f"{training_dir_path}/input/{idx_cg}.pdb"

            # Save current stack to file
            with open(save_path, "w") as cg_file:
                for line in line_stack_cg:
                    cg_file.write(line)

            # Save box size in list
            cg_box_size_dataset_relations.append((idx_cg, cg_box_size))

            # Fix PBC of the coordinates
            coordinates_cg = np.array(coordinates_cg)
            coordinates_cg = coordinates_cg - np.floor(coordinates_cg / cg_box_size) * cg_box_size

            # Calculate COM of residue
            com = np.mean(coordinates_cg, axis=0)
            cg_com_list.append((com, idx_cg))

            # Reset stack and residue index
            line_stack_cg = []
            last_residue_idx_cg = residue_idx_cg
            coordinates_cg = []

            idx_cg += 1

        # Add the line to the stack
        line_stack_cg.append(cg_line)

    # Loop over the lines of the AT files
    for j, at_line in enumerate(at_lines):
        # Check if we have reached the max number of samples
        if idx_at > MAX_SAMPLES:
            break

        # Split the lines and get the residue index
        at_line_split = at_line.split()
        residue_idx_at = int(at_line_split[4]) - 1  # <- ATOM lines start at 1, CG lines start at 0. I don't know why

        # Check if the residue changes or last line
        if residue_idx_at != last_residue_idx_at or j == len(at_lines) - 1:
            # Add the line to the stack
            if j == len(at_lines) - 1:
                line_stack_at.append(at_line)

            # Make folder for the residue if not in training data mode
            if not TRAINING_DATA_MODE:
                if not os.path.exists(f"{output_dir_path}/{i}_{last_residue_idx_at}"):
                    os.makedirs(f"{output_dir_path}/{i}_{last_residue_idx_at}")

            # Add TER line to the stack
            line_stack_at.append(f"TER     {(139 + 138*last_residue_idx_at) % 100000}      DOPC    {last_residue_idx_at + 1}\n")  # <- I don't know why this is needed, but biopython does it too
            line_stack_at.append("END   ")

            # Save also to training data
            save_path = f"{output_dir_path}/{i}_{last_residue_idx_at}/at.pdb" if not TRAINING_DATA_MODE else f"{training_dir_path}/output/{idx_at}.pdb"

            # Save current stack to file
            with open(save_path, "w") as at_file:
                for line in line_stack_at:
                    at_file.write(line)

            # Save box size in list
            at_box_size_dataset_relations.append((idx_at, at_box_size))

            # Reset stack and residue index
            line_stack_at = []
            last_residue_idx_at = residue_idx_at

            # Print progress
            if idx_at % 100 == 0:
                timestamp = datetime.fromtimestamp(
                    time.time()).strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"[{timestamp}] Generated {idx_at} training samples ({elapsed_time()})")

            idx_at += 1

        # Add the line to the stack
        line_stack_at.append(at_line)

    # Find all neighbours of each CG residue
    for entry in cg_com_list:
        # Get all neighbours of the current residue
        neighbours = [entry_t[1] for entry_t in cg_com_list if entry_t[1] != entry[1] and np.linalg.norm(entry[0] - entry_t[0]) < NEIGHBOUR_CUTOFF_RADIUS]

        # Add to list
        neighbours_counts.append(len(neighbours))
        neighbourhoods.append(neighbours)

    if idx_cg > MAX_SAMPLES and idx_at > MAX_SAMPLES:
        break


# Save box size dataset relations as csv
path = os.path.join(output_box_table_path, "box_sizes_cg.csv")
with open(path, "a") as file:
    for entry in cg_box_size_dataset_relations:
        file.write(f"{entry[1][0]},{entry[1][1]},{entry[1][2]}\n")

path = os.path.join(output_box_table_path, "box_sizes_at.csv")
with open(path, "a") as file:
    for entry in at_box_size_dataset_relations:
        file.write(f"{entry[1][0]},{entry[1][1]},{entry[1][2]}\n")
    
path = os.path.join(output_box_table_path, "neighborhoods.csv")
with open(path, "a") as file:
    for list in neighbourhoods:
        file.write(",".join([str(l) for l in list]) + "\n")

# Calculate mean and std of neighbours counts
neighbours_counts = np.array(neighbours_counts)
neighbours_counts_mean = np.mean(neighbours_counts)
neighbours_counts_std = np.std(neighbours_counts)
neighbours_counts_max = np.max(neighbours_counts)
neighbours_counts_min = np.min(neighbours_counts)

# Print
print(f"Neighbour count: {neighbours_counts_mean} +- {neighbours_counts_std}, max: {neighbours_counts_max}, min: {neighbours_counts_min}")

# Make histogram of neighbours counts and plot it
plt.hist(neighbours_counts, bins=100)

# Plot a line for the target neighbour count
plt.axvline(x=TARGET_NEIGHBOUR_COUNT, color='r', linestyle='dashed', linewidth=1)

# Plot mean and std
plt.axvline(x=neighbours_counts_mean, color='g', linestyle='dashed', linewidth=1)
plt.axvline(x=neighbours_counts_mean + neighbours_counts_std, color='g', linestyle='dashed', linewidth=1)
plt.axvline(x=neighbours_counts_mean - neighbours_counts_std, color='g', linestyle='dashed', linewidth=1)

# Add percentage of residues that have less than TARGET_NEIGHBOUR_COUNT neighbours
plt.text(0.1, 0.65, f"{np.sum(neighbours_counts <= TARGET_NEIGHBOUR_COUNT) / len(neighbours_counts) * 100:.2f}%", transform=plt.gca().transAxes)
# Add percentage of residues that have more than TARGET_NEIGHBOUR_COUNT neighbours
plt.text(0.75, 0.65, f"{np.sum(neighbours_counts > TARGET_NEIGHBOUR_COUNT) / len(neighbours_counts) * 100:.2f}%", transform=plt.gca().transAxes)

# Add legend
plt.legend(["Target neighbour count", "Mean", "Std"])

plt.title(f"Histogram of neighbours counts (cuoff {NEIGHBOUR_CUTOFF_RADIUS} A)")
plt.xlabel("Neighbours count")
plt.ylabel("Frequency")
plt.show()
