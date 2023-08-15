import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
import sys

from library.parser import get_cg_at_datasets
from library.static.topologies import DOPC_CG_NAME_TO_TYPE_MAP, DOPC_BEAD_TYPE_NAME_IDS, DOPC_ELEMENT_TYPE_NAME_IDS
from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from library.static.vector_mappings import DOPC_AT_MAPPING, DOPC_CG_MAPPING
import matplotlib.pyplot as plt

PADDING_X = 3 # Padding on left and right
PADDING_Y = 1 # Padding on top and bottom

SIMULATION_BOX_SIZE = (191.279,  191.279,  111.367)  # In angstroms
PBC_CUTOFF = 10.0  # If a bond is longer than this, it is considered to be on the other side of the simulation box
BOX_SCALE_FACTOR = 100.0  # The scale factor to downscale the bond vectors with

def fix_pbc(vector):
    """
        Fixes the periodic boundary conditions of a vector.
        :param vector: The vector to fix
        :return: The fixed vector
    """
    if vector[0] > PBC_CUTOFF:
        vector[0] -= SIMULATION_BOX_SIZE[0]
    elif vector[0] < -PBC_CUTOFF:
        vector[0] += SIMULATION_BOX_SIZE[0]
    if vector[1] > PBC_CUTOFF:
        vector[1] -= SIMULATION_BOX_SIZE[1]
    elif vector[1] < -PBC_CUTOFF:
        vector[1] += SIMULATION_BOX_SIZE[1]
    if vector[2] > PBC_CUTOFF:
        vector[2] -= SIMULATION_BOX_SIZE[2]
    elif vector[2] < -PBC_CUTOFF:
        vector[2] += SIMULATION_BOX_SIZE[2]

    if vector.norm() > PBC_CUTOFF:
        return fix_pbc(vector)
    else:
        return vector

def is_output_matrix_healthy(output):
    healthy = True
    # Check if values that are not in [-1, 1] exist
    if np.max(output) > 1 or np.min(output) < -1:
            healthy = False
    # Check for nan or inf
    if np.isnan(output).any() or np.isinf(output).any():
        healthy = False
        
    # Diagnostic
    if not healthy:
        # Find how many values are outside of [-1, 1]
        print(f"Found {np.sum(output > 1) + np.sum(output < -1)} values outside of [-1, 1]!")
        
        # for i, x in enumerate(output):
        #     if np.max(x) > 1 or np.min(x) < -1:
        #         # Save matrix as image
        #         plt.imshow(x[:, :, 0])
        #         plt.colorbar()
        #         plt.savefig(f"data/figures/failing_matrix_{i}.png")
        #         plt.clf()
    
    return healthy

class RelativeVectorsTrainingDataGenerator(tf.keras.utils.Sequence):
    """
        A data generator class that generates batches of data for the CNN.
        The input and output is the relative vectors between the atoms and the beads.
    """

    def __init__(
        self, 
        input_dir_path, 
        output_dir_path, 
        input_size=(11 + 2 * PADDING_X, 3 + 2 * PADDING_Y, 1),
        output_size=(53 + 2 * PADDING_X, 3 + 2 * PADDING_Y, 1),
        shuffle=False, 
        batch_size=1,
        validate_split=0.1,
        validation_mode=False,
        ):
        """
            Initializes a data generator object to generate batches of data.
            :param input_dir_path: The path to the directory where the input data (X) is located
            :param output_dir_path: The path to the directory where the output data (Y) is located
            :param input_size: The size/shape of the input data
            :param output_size: The size/shape of the output data
            :param shuffle: If the data should be shuffled after each epoch
            :param batch_size: The size of each batch
            :param validate_split: The percentage of data that should be used for validation
            :param validation_mode: If the generator should be in validation mode
        """
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        self.input_size = input_size
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.validation_mode = validation_mode
        
        self.parser = PDBParser(QUIET=True)
        
        max_index = int(os.listdir(input_dir_path).__len__() - 1)

        self.len = int(max_index * (1 - validate_split))
        self.start_index = 0
        self.end_index = self.len

        # Set validation mode
        if self.validation_mode:
            self.start_index = self.len + 1
            self.len = int(max_index * validate_split)
            self.end_index = max_index
            
        # Debug
        print(f"Found {self.len} residues in ({self.input_dir_path})")

        # Initialize
        self.on_epoch_end()

    def on_epoch_end(self):
        pass

    def __len__(self):
        return self.len // self.batch_size

    def __getitem__(self, idx):
        # Initialize Batch
        X = np.zeros((self.batch_size, *self.input_size), dtype=np.float32)
        Y = np.zeros((self.batch_size, *self.output_size), dtype=np.float32)
        # print(X)
        # print(f"EMPTY MATRIX: Max: {np.max(X)}, min: {np.min(X)}")
        # print(f"EMPTY MATRIX: Max: {np.max(Y)}, min: {np.min(Y)}")

        # Load the two files in the idx folder
        for i in range(self.batch_size):
            # Get the index of the residue
            residue_idx = idx * self.batch_size + i

            # Check if end index is reached
            if self.validation_mode and residue_idx > self.end_index:
                raise Exception(f"You are trying to access a residue that does not exist ({residue_idx})!")

            # Get the path to the files
            cg_path = f"{self.input_dir_path}/{residue_idx}.pdb"
            at_path = f"{self.output_dir_path}/{residue_idx}.pdb"

            # Load the files
            cg_structure = self.parser.get_structure(residue_idx, cg_path)
            at_structure = self.parser.get_structure(residue_idx, at_path)

            # Get the residues
            cg_residue = list(cg_structure.get_residues())[0]
            at_residue = list(at_structure.get_residues())[0]

            # Get the atoms
            cg_atoms = list(cg_residue.get_atoms())
            at_atoms = list(at_residue.get_atoms())
            
            # Remove the hydrogen atoms 
            # at_atoms = [atom for atom in at_atoms if atom.element != "H"] # Currently not used because we use a dict
            
            # Make name -> atom dict
            cg_atoms_dict = {atom.get_name(): atom for atom in cg_atoms}
            at_atoms_dict = {atom.get_name(): atom for atom in at_atoms}
            

            # Make the cg relative vectors
            for j, (at1_name, at2_name) in enumerate(DOPC_CG_MAPPING):
                if not (cg_atoms_dict.get(at1_name) and cg_atoms_dict.get(at2_name)):
                    raise Exception(f"Missing {at1_name} or {at2_name} in residue {residue_idx}")
                else:
                    rel_vector = cg_atoms_dict.get(at2_name).get_vector() - cg_atoms_dict.get(at1_name).get_vector()
                    
                    # Fix the PBC
                    if rel_vector.norm() > PBC_CUTOFF:
                        rel_vector = fix_pbc(rel_vector)
                        
                        # Consitency check
                    if rel_vector.norm() > 3 * PBC_CUTOFF:
                        raise Exception(
                            f"Found a very big vector in res {residue_idx} ({rel_vector})!")

                    # Check if nan or inf is in the vector
                    if np.isnan(rel_vector[0]) or np.isnan(rel_vector[1]) or np.isnan(rel_vector[2]) or np.isinf(rel_vector[0]) or np.isinf(rel_vector[1]) or np.isinf(rel_vector[2]):
                        raise Exception(
                            f"Found nan or inf in residue {residue_idx} ({rel_vector})!")

                    
                    X[i, j + PADDING_X, PADDING_Y + 0, 0] = rel_vector[0] / BOX_SCALE_FACTOR
                    X[i, j + PADDING_X, PADDING_Y + 1, 0] = rel_vector[1] / BOX_SCALE_FACTOR
                    X[i, j + PADDING_X, PADDING_Y + 2, 0] = rel_vector[2] / BOX_SCALE_FACTOR

                    # if not is_output_matrix_healthy(X):
                    #     print(rel_vector[0] / BOX_SCALE_FACTOR)
                    #     print(rel_vector[1] / BOX_SCALE_FACTOR)
                    #     print(rel_vector[2] / BOX_SCALE_FACTOR)
                    #     print(X[i, :, :, 0])
                    #     print(f"X: {np.max(X)}, {np.min(X)}, healthy: {is_output_matrix_healthy(X)}, res: {residue_idx}, vec: {rel_vector}")

            # Make the relative vectors out of a vector mapping
            for j, (at1_name, at2_name) in enumerate(DOPC_AT_MAPPING):
                if not (at_atoms_dict.get(at1_name) and at_atoms_dict.get(at2_name)):
                    raise Exception(f"Missing {at1_name} or {at2_name} in residue {residue_idx}")
                else:
                    rel_vector = at_atoms_dict.get(at2_name).get_vector() - at_atoms_dict.get(at1_name).get_vector()
                    
                    # Fix the PBC
                    if rel_vector.norm() > PBC_CUTOFF:
                        rel_vector = fix_pbc(rel_vector)

                    # Consitency check
                    if rel_vector.norm() > 3 * PBC_CUTOFF:
                        raise Exception(f"Found a very big vector in res {residue_idx} ({rel_vector})!")
                        
                    # Check if nan or inf is in the vector
                    if np.isnan(rel_vector[0]) or np.isnan(rel_vector[1]) or np.isnan(rel_vector[2]) or np.isinf(rel_vector[0]) or np.isinf(rel_vector[1]) or np.isinf(rel_vector[2]):
                        raise Exception(
                            f"Found nan or inf in residue {residue_idx} ({rel_vector})!")
                    
                    Y[i, j + PADDING_X, PADDING_Y + 0, 0] = rel_vector[0] / BOX_SCALE_FACTOR
                    Y[i, j + PADDING_X, PADDING_Y + 1, 0] = rel_vector[1] / BOX_SCALE_FACTOR
                    Y[i, j + PADDING_X, PADDING_Y + 2, 0] = rel_vector[2] / BOX_SCALE_FACTOR

        # Convert to tensor
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        # Check if values that are not in [-1, 1] exist
        if not is_output_matrix_healthy(Y) or not is_output_matrix_healthy(X):
            # print(f"X: {np.max(X)}, {np.min(X)}, healthy: {is_output_matrix_healthy(X)}")
            # print(f"Y: {np.max(Y)}, {np.min(Y)}, healthy: {is_output_matrix_healthy(Y)}")
            raise Exception(f"Found values outside of [-1, 1], see print before.")

        # Return tensor as deep copy
        return tf.identity(X), tf.identity(Y)
