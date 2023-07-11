from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.NeighborSearch import NeighborSearch

import matplotlib.pyplot as plt
import numpy as np


def plot_residue(residue: Residue, residue_map=None, bond_map=None):
    """
        Plots the residue using matplotlib

        Parameters:
            residue (Residue): The residue to plot
            map (dict): A dictionary mapping atom names to type
            bond_map (dict): A set mapping atom i to atom j if they are bonded
    """

    if not isinstance(residue, Residue):
        raise TypeError('residue must be of type Residue')

    # Get the coordinates of all atoms in the residue
    coordinates = np.array([atom.get_coord() for atom in residue.get_atoms()])
    x, y, z = coordinates.T

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Dict for random colors for each atom type
    color_map = {}

    if residue_map is None:
        # Create a random color for each atom type
        for atom in residue.get_atoms():
            print(atom.get_name())
            if atom.get_name() not in color_map.keys():
                color_map[atom.get_name()] = np.random.rand(3,)
    else:
        # Use the given map
        for type in residue_map.keys():
            if residue_map[type] not in color_map.keys():
                color_map[residue_map[type]] = np.random.rand(3,)

    # Color the atoms by their element
    colors = [color_map[atom.get_name()] for atom in residue.get_atoms()] if residue_map is None else [
        color_map[residue_map[atom.get_name()]] for atom in residue.get_atoms()]

    # Add legend
    ax.legend(handles=[plt.Line2D([0], [0], color=color, linewidth=3, linestyle='-')
              for color in color_map.values()], labels=list(color_map.keys()))

    # Plot the coordinates
    ax.scatter(x, y, z, c=colors)

    # Add lines between the atoms that are bonded using the bond_map dict i -> j
    if bond_map is not None:
        for i in bond_map.keys():
            ax.plot([x[i - 1], x[bond_map[i] - 1]], 
                    [y[i - 1], y[bond_map[i] - 1]],
                    [z[i - 1], z[bond_map[i] - 1]], 
                    color='black')
    plt.show()
