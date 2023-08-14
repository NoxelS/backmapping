from Bio.PDB.Atom import Atom
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBParser import PDBParser

from library.classes.Dataset import Dataset
from library.parser import get_cg_at_datasets
from library.static.utils import DEFAULT_ELEMENT_COLOR_MAP
from library.static.topologies import cg_name_to_type_dict, cg_bond_map_dict

import os
import matplotlib.pyplot as plt
import numpy as np

from library.viz import plot_residue

# Load residue
cg_struct = PDBParser().get_structure("DOPC", "data/training/0/cg.pdb")
at_struct = PDBParser().get_structure("DOPC", "data/results/test_2/at_0.pdb")
at_struct_real = PDBParser().get_structure("DOPC", "data/training/0/at.pdb")

cg_residue = list(cg_struct.get_residues())[0]
at_residue = list(at_struct.get_residues())[0]
at_residue_real = list(at_struct_real.get_residues())[0]

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


# Plot the figures side by side
fig = plt.figure(figsize=(15, 10))
ax0 = fig.add_subplot(131, projection='3d')
ax1 = fig.add_subplot(132, projection='3d')
ax2 = fig.add_subplot(133, projection='3d')


plot_residue(cg_residue,
             residue_map=cg_residue_map, bond_map=cg_bond_map, show_labels=True, group_by_element=True, dont_show_plot=True, ax=ax0, fig=fig)

plot_residue(at_residue, residue_map=at_residue_map,
             bond_map=at_bond_map, show_labels=False, group_by_element=True, dont_show_plot=True, ax=ax1, fig=fig)

plot_residue(at_residue_real, residue_map=at_residue_map,
             bond_map=at_bond_map, show_labels=False, group_by_element=True, dont_show_plot=True, ax=ax2, fig=fig)

# Add title
fig.suptitle(f'CG vs Predicted vs Expected', fontsize=16)

# Thight layout
fig.tight_layout()

# Plot
plt.show()