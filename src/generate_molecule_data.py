import os
import time
from datetime import datetime

from Bio.PDB.PDBIO import PDBIO

from library.parser import get_cg_at_datasets

input_dir_path = "data/membranes"   # TODO: make with config
output_dir_path = "data/molecules_test"
cg_datasets, at_datasets = [], []
io = PDBIO()
idx = 0

# Get all CG and AT datasets
cg_datasets, at_datasets = get_cg_at_datasets(input_dir_path, CG_PATTERN='cg.pdb', AT_PATTERN='at.pdb')

# Create output dir if it does not exist
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# Find last idx in output dir if it exists
last_membrane, last_idx = 0, 0
if os.listdir(output_dir_path):
    last_membrane, last_idx = [f for f in os.listdir(output_dir_path)][-1].split("_")

# Loop over both at the same time (these are generators, so they are not loaded into memory immediately)
for i, (cg_dataset, at_dataset) in enumerate(zip(cg_datasets, at_datasets)):
    for j, (cg_residue, at_residue) in enumerate(zip(cg_dataset.get_residues(), at_dataset.get_residues())):
        # Skip if we already processed this one
        if idx < int(last_idx):
            idx += 1
            continue
        else:
            if last_idx != 0:
                print(f"Starting at membrane {last_membrane} and residue {last_idx}")
            last_idx = 0

        # Create folder for the idx
        folder_name = f"{i}_{idx}"  # <membrane_idx>_<residue_idx>
        if not os.path.exists(f"{output_dir_path}/{folder_name}"):
            os.makedirs(f"{output_dir_path}/{folder_name}")

        # io.set_structure(cg_dataset.get_structure())
        # io.save(f"{output_dir_path}/{folder_name}/cg.pdb",
        #         ResidueSelector(cg_residue._id[1]), preserve_atom_numbering=True)

        # io.set_structure(at_dataset.get_structure())
        # io.save(f"{output_dir_path}/{folder_name}/at.pdb",
        #         ResidueSelector(at_residue._id[1]), preserve_atom_numbering=True)

        # Print progress
        if idx % 10 == 0:
            timestamp = datetime.fromtimestamp(
                time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Generated {idx} training samples")

        idx += 1
