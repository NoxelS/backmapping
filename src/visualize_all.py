import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB.PDBParser import PDBParser

from library.classes.Dataset import Dataset
from library.Visualization import plot_residue
from library.DataParser import get_cg_at_datasets
from library.static.MartiniMaps import cg_name_to_type_dict, cg_bond_map_dict

# Get all CG and AT datasets
cg_datasets, at_datasets = get_cg_at_datasets(os.path.join("data", "raw"))

# Bond map for cg system
cg_residue_map = cg_name_to_type_dict(os.path.join(
    "data", "topologies", "martini_v2.0_DOPC_02.itp"))
cg_bond_map = cg_bond_map_dict(os.path.join(
    "data", "topologies", "martini_v2.0_DOPC_02.itp"))

# Bond map for at system
at_residue_map = cg_name_to_type_dict(os.path.join(
    "data", "raw", "CG2AT_2023-02-13_20-20-52", "FINAL", "DOPC.itp"))
at_bond_map = cg_bond_map_dict(os.path.join(
    "data", "raw", "CG2AT_2023-02-13_20-20-52", "FINAL", "DOPC.itp"))

# Loop over both at the same time
for cg_dataset, at_dataset in zip(cg_datasets, at_datasets):

    # Create Folder if it does not exist
    if not os.path.exists(os.path.join("data", "figures", "raw_data", cg_dataset.parent)):
        os.makedirs(os.path.join("data", "figures",
                                 "raw_data", cg_dataset.parent))
        
    for i, (cg_residue, at_residue) in enumerate(zip(cg_dataset.get_residues(), at_dataset.get_residues())):

        # Plot the figures side by side
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        plot_residue(cg_residue,
                     residue_map=cg_residue_map, bond_map=cg_bond_map, show_labels=True, group_by_element=True, dont_show_plot=True, ax=ax1, fig=fig)

        plot_residue(at_residue, residue_map=at_residue_map,
                     bond_map=at_bond_map, show_labels=False, group_by_element=True, dont_show_plot=True, ax=ax2, fig=fig)

        # Add title
        fig.suptitle(f'CG vs AT ({cg_dataset.parent})', fontsize=16)

        # Save the figure
        plt.savefig(os.path.join("data", "figures",
                    "raw_data", cg_dataset.parent, f"{i}.png"))

    print(f"Plotted {cg_dataset.parent}")
