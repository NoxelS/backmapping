from Bio.PDB.Atom import Atom
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.NeighborSearch import NeighborSearch

from library.classes.Dataset import Dataset
from library.DataParser import get_cg_at_datasets
from library.static.ColorMaps import DEFAULT_ELEMENT_COLOR_MAP
from library.static.MartiniMaps import cg_name_to_type_dict, cg_bond_map_dict

import os
import matplotlib.pyplot as plt
import numpy as np


def __get_name(atom: Atom):
    return atom.get_name()


def __get_element(atom: Atom):
    return atom.element


def plot_residue(
    residue: Residue,
    residue_map=None,
    bond_map=None,
    show_labels=False,
    group_by_element=False,
    show_neighrbor_bonds=False,
    neighbor_distance=0.5,
    save_path=None,
    dont_show_plot=False,
    ax=None,
    fig=None
) -> plt.Figure:
    """
        Plots the residue using matplotlib

        Parameters:
            residue (Residue): The residue to plot
            map (dict): A dictionary mapping atom names to type
            bond_map (dict): A set mapping atom i to atom j if they are bonded
            show_labels (bool): If True, the atoms will be labeled
            group_by_element (bool): If True, the atoms will be colored by their element
            show_neighrbor_bonds (bool): If True, the bonds (estimated) will be shown
            neighbor_distance (float): The distance to search for neighbors
            save_path (str): The path to save the plot to, will not show the plot if given
            dont_show_plot (bool): If True, the plot will not be shown
            ax (matplotlib.axes.Axes): The axes to plot on (must be given if fig is given)
            fig (matplotlib.figure.Figure): The figure to plot on (must be given if ax is given)

        Returns:
            fig (matplotlib.figure.Figure): The figure object
    """

    if not isinstance(residue, Residue):
        raise TypeError('residue must be of type Residue')

    atom_id = __get_element if group_by_element else __get_name

    # Get the coordinates of all atoms in the residue
    coordinates = np.array([atom.get_coord() for atom in residue.get_atoms()])
    x, y, z = coordinates.T

    # Create the figure
    if ax is None and fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Dict for random colors for each atom type
    color_map = {}

    if residue_map is None or group_by_element:
        # Create a random color for each atom type
        for atom in residue.get_atoms():
            if atom_id(atom) not in color_map.keys():
                if atom.element in DEFAULT_ELEMENT_COLOR_MAP.keys():
                    color_map[atom_id(
                        atom)] = DEFAULT_ELEMENT_COLOR_MAP[atom.element]
                else:
                    color_map[atom_id(atom)] = np.random.rand(3,)
    else:
        # Use the given map
        for type in residue_map.keys():
            if residue_map[type] not in color_map.keys():
                color_map[residue_map[type]] = np.random.rand(3,)

    # Color the atoms by their element
    colors = []

    if residue_map is None:
        colors = [color_map[atom_id(atom)] for atom in residue.get_atoms()]
    elif group_by_element:
        if residue_map is not None:
            for atom in residue.get_atoms():
                if atom.element in DEFAULT_ELEMENT_COLOR_MAP.keys():
                    colors.append(DEFAULT_ELEMENT_COLOR_MAP[atom.element])
                else:
                    colors.append(color_map[atom_id(atom)])
        else:
            colors = [color_map[atom_id(atom)] for atom in residue.get_atoms()]
    else:
        colors = [color_map[residue_map[atom.get_name()]]
                  for atom in residue.get_atoms()]

    # Add legend
    ax.legend(handles=[plt.Line2D([0], [0], color=color, linewidth=3, linestyle='-')
              for color in color_map.values()
                       ], labels=list(color_map.keys()))

    # Plot the coordinates
    ax.scatter(x, y, z, c=colors, edgecolors='black')

    # Add lines between the atoms that are bonded using the bond_map dict i -> j
    if bond_map is not None:
        for i in bond_map.keys():
            if isinstance(bond_map[i], list):
                for j in bond_map[i]:
                    at_from = list(residue.get_atoms())[i - 1]
                    at_to = list(residue.get_atoms())[j - 1]
                    ax.plot([at_from.get_coord()[0], at_to.get_coord()[0]], [
                            at_from.get_coord()[1], at_to.get_coord()[1]], [at_from.get_coord()[2], at_to.get_coord()[2]],
                            color='black', linestyle='--', linewidth=1)
            else:
                at_from = list(residue.get_atoms())[i - 1]
                at_to = list(residue.get_atoms())[bond_map[i] - 1]
                ax.plot([at_from.get_coord()[0], at_to.get_coord()[0]], [
                        at_from.get_coord()[1], at_to.get_coord()[1]], [at_from.get_coord()[2], at_to.get_coord()[2]],
                        color='black', linestyle='--', linewidth=1)

    elif show_neighrbor_bonds:
        ns = NeighborSearch(list(residue.get_atoms()))
        for atom in residue.get_atoms():
            neighbors = ns.search(atom.get_coord(), neighbor_distance)
            for neighbor in neighbors:
                ax.plot([atom.get_coord()[0], neighbor.get_coord()[0]], [
                        atom.get_coord()[1], neighbor.get_coord()[1]], [atom.get_coord()[2], neighbor.get_coord()[2]],
                        color='black', linestyle='--', linewidth=1)

    # Add labels to the atoms if show_labels is True
    if show_labels:
        if residue_map is None or group_by_element:
            for i, atom in enumerate(residue.get_atoms()):
                ax.text(atom.get_coord()[0], atom.get_coord()[
                        1], atom.get_coord()[2], atom_id(atom))
        else:
            for i, atom in enumerate(residue.get_atoms()):
                ax.text(atom.get_coord()[0], atom.get_coord()[
                        1], atom.get_coord()[2], residue_map[atom_id(atom)])

    # Save the plot
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    if not dont_show_plot:
        plt.show()

    return ax


def show_dataset(
    name: str,
    residue_index: int,
    dont_show_plot=False,
):
    # Get all CG and AT datasets
    cg_datasets, at_datasets = get_cg_at_datasets(os.path.join("data", "raw"))
    cg_dataset, at_dataset = None, None

    # Find the dataset with the given name
    for dataset in cg_datasets:
        if dataset.parent == name:
            cg_dataset = dataset
            break
    else:
        raise ValueError(f"Could not find dataset with name {name}")
    
    for dataset in at_datasets:
        if dataset.parent == name:
            at_dataset = dataset
            break
    else:
        raise ValueError(f"Could not find dataset with name {name}")

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

    # Create Folder if it does not exist
    if not os.path.exists(os.path.join("data", "figures", "raw_data", cg_dataset.parent)):
        os.makedirs(os.path.join("data", "figures",
                                    "raw_data", cg_dataset.parent))

    for i, (cg_residue, at_residue) in enumerate(zip(cg_dataset.get_residues(), at_dataset.get_residues())):
        if i == residue_index:
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
            if not dont_show_plot:
                plt.show()
            else:
                plt.savefig(os.path.join(f"{cg_dataset.parent}-{i}.png"))
                print(f"Saved figure to {os.path.join(f'{cg_dataset.parent}-{i}.png')}")
