import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from library.config import Keys, config
from library.parser import get_cg_at_datasets

DATA_PREFIX = config(Keys.DATA_PATH)

input_dir_path = os.path.join(DATA_PREFIX, "membranes")
output_dir_path = os.path.join(DATA_PREFIX, "molecules")
training_dir_path = os.path.join(DATA_PREFIX, "training")
output_box_table_path = DATA_PREFIX  # <- This is the path where the box sizes are saved

TRAINING_DATA_MODE = True
TARGET_NEIGHBOUR_COUNT = 4        # The target number of neighbours for each residue, only used for logging
NEIGHBOUR_CUTOFF_RADIUS = 10.0     # The radius in which the neighbours are considered (in Angstrom)

def generate_molecule_data(max_samples = 15 * (10 + 1) * 1024):
    """
    This function generates training data for the backmapping algorithm. It reads in coarse-grained (CG) and atomistic (AT) 
    structures from input_dir_path and generates training data by creating an AT structure for each 
    residue in the CG structure. The AT structure is created by copying the AT structure of the corresponding residue in the 
    AT structure. The AT structure is then saved to output_dir_path. If TRAINING_DATA_MODE is True, the AT structures are 
    saved to training_dir_path instead. The function also calculates the mean distance to the N atom for each residue and 
    the number of neighbours within a certain radius for each residue. The results are saved to csv files in output_box_table_path.
    """

    cg_datasets, at_datasets = [], []
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

    coordinates_at = []  # Only used to calculate mean distance to N for debugging

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
            if idx_cg > max_samples:
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
            if idx_at > max_samples:
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

            # Get coordinates of the atom and add to dict
            aton_name = at_line_split[2]
            coordinates = (float(at_line_split[5]), float(at_line_split[6]), float(at_line_split[7]))
            if coordinates_at.__len__() <= idx_at:
                coordinates_at.append({})

            coordinates_at[idx_at][aton_name] = coordinates

            # Add the line to the stack
            line_stack_at.append(at_line)

        # Find all neighbours of each CG residue
        for entry in cg_com_list:
            # Get all neighbours of the current residue
            neighbours = [entry_t[1] for entry_t in cg_com_list if entry_t[1] != entry[1] and np.linalg.norm(entry[0] - entry_t[0]) < NEIGHBOUR_CUTOFF_RADIUS]

            # Add to list
            neighbours_counts.append(len(neighbours))
            neighbourhoods.append(neighbours)

        if idx_cg > max_samples and idx_at > max_samples:
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


    # Calculate the mean distance to the N atom for each residue
    mean_distances = {}  # Entry is (atom_name, mean_distance)

    for i, molecule_dict in enumerate(coordinates_at):
        for atom_name, coordinates in molecule_dict.items():
            if atom_name == "N":
                continue

            if atom_name not in mean_distances:
                mean_distances[atom_name] = []

            # Check if N is in molecule_dict
            if "N" in molecule_dict:
                distance = np.linalg.norm(np.array(coordinates) - np.array(molecule_dict["N"]))

                # Fix PBC if distance is larger than half the box size
                box = np.array(at_box_size_dataset_relations[i][1])
                distance = np.min([distance, np.linalg.norm(distance - box)])

                mean_distances[atom_name].append(distance)

    # Calculate mean and std
    mean_distances_mean = {}
    mean_distances_std = {}
    for atom_name, distances in mean_distances.items():
        mean_distances_mean[atom_name] = np.mean(distances)
        mean_distances_std[atom_name] = np.std(distances)


    # Write the mean distances to a csv file
    path = os.path.join(output_box_table_path, "mean_distances.csv")
    with open(path, "a") as file:
        file.write("atom_name,mean,std\n")
        for atom_name, mean in mean_distances_mean.items():
            file.write(f"{atom_name},{mean},{mean_distances_std[atom_name]}\n")

    ####### PLOT HISTOGRAM #######


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


if __name__ == "__main__":
    generate_molecule_data()