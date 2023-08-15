import os
from library.classes.dataset import Dataset
from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Structure, Residue, Atom, Model, Chain
import tensorflow as tf
from library.static.topologies import DOPC_CG_NAME_TO_TYPE_MAP, DOPC_BEAD_TYPE_NAME_IDS, DOPC_ELEMENT_TYPE_NAME_IDS
import numpy as np
import time
from datetime import datetime
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def find_all_pdb_files(path):
    """
        Find all pdb files recursivly in a given path and return a list of Dataset objects
    """
    pdb_files = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.pdb'):
                pdb_files.append(Dataset(entry.name, entry.path, 'pdb'))
            elif entry.is_dir():
                pdb_files.extend(find_all_pdb_files(entry.path))

    return pdb_files


def get_pdb_file_paths_dic(path):
    """
        Returns a dictionary of pdb datasets where the key is the name of the pdb files folder
        E.g. {'CG2AT_2023-02-13_20-20-52': [<Dataset object at 0xa>, ...']}
    """
    pdb_files_dic = {}
    with os.scandir(path) as data_folders:
        for data_folder in [d for d in data_folders if not d.is_file()]:
            # Find all pdb files in the data folder
            datasets = find_all_pdb_files(data_folder.path)

            # Add parent to datasets so we know where they came from
            for dataset in datasets:
                dataset.parent = data_folder.name

            # Add the datasets to the dictionary
            pdb_files_dic[data_folder.name] = datasets

    return pdb_files_dic


def get_cg_at_datasets(
        path,
        CG_PATTERN='CG_INPUT.pdb',
        AT_PATTERN='final_cg2at_de_novo.pdb'
):
    """
        Get all CG and AT datasets from a given path.
        This uses the folder structure provided by chetan.
        E.g data/raw/CG2AT_2023-02-13_20-20-52/
            /FINAL/final_cg2at_de_novo.pdb
            /INPUT/CG_INPUT.pdb
            /INPUT/DOPC_Frame_....pdb
            /MERGED/merged_cg2at_de_novo.pdb
            ...

        Parameters:
            path (str): The path to the data folder
            CG_RELATIVE_PATH (str): The pattern to match for CG pdb files
            AT_RELATIVE_PATH (str): The pattern to match for AT pdb files

        Returns:
            cg_datasets (list): A list of CG datasets
            at_datasets (list): A list of AT datasets
    """

    all_pdb_files_dic = get_pdb_file_paths_dic(path)

    cg_datasets = []
    at_datasets = []

    for key in all_pdb_files_dic.keys():
        for dataset in all_pdb_files_dic[key]:
            if dataset.path.endswith(CG_PATTERN):
                cg_datasets.append(dataset)
            elif dataset.path.endswith(AT_PATTERN):
                at_datasets.append(dataset)

    return cg_datasets, at_datasets


def get_structure_from_dataset(dataset):
    """
        Returns a Bio.PDB.Structure object from a given dataset
    """
    parser = PDBParser()
    return parser.get_structure(dataset.name, dataset.path)


class ResidueSelector(Select):
    def __init__(self, target_id):
        self.target_id = target_id

    def accept_residue(self, residue):
        # TODO: improve this
        return residue._id[1] == self.target_id

def generate_training_data(path_to_raw_data, output_dir_path):
    # Get all CG and AT datasets (this is only indexing the data, not loading it)
    cg_datasets, at_datasets = get_cg_at_datasets(path_to_raw_data)

    io = PDBIO()
    idx = 0

    # Loop over both at the same time (these are generators, so they are not loaded into memory immediately)
    for i, (cg_dataset, at_dataset) in enumerate(zip(cg_datasets, at_datasets)):
        for j, (cg_residue, at_residue) in enumerate(zip(cg_dataset.get_residues(), at_dataset.get_residues())):
            # Create folder for the idx
            if not os.path.exists(f"{output_dir_path}/{idx}"):
                os.makedirs(f"{output_dir_path}/{idx}")

            io.set_structure(cg_dataset.get_structure())
            io.save(f"{output_dir_path}/{idx}/cg.pdb",
                    ResidueSelector(cg_residue._id[1]), preserve_atom_numbering=True)

            io.set_structure(at_dataset.get_structure())
            io.save(f"{output_dir_path}/{idx}/at.pdb",
                    ResidueSelector(at_residue._id[1]), preserve_atom_numbering=True)

            idx += 1

            if idx % 10 == 0:
                timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Generated {idx} training examples")


def pdb_data_to_xyz(batch_size, idx, input_dir_path, input_size, output_size):
    """
        Converts a pdb file to a xyz file
    """      

    # Initialize Batch
    X = np.empty((batch_size, *input_size))
    Y = np.empty((batch_size, *output_size))

    # Load the two files in the idx folder
    for i in range(batch_size):
        # Get the index of the residue
        residue_idx = idx * batch_size + i

        # Get the path to the files
        cg_path = f"{input_dir_path}/{residue_idx}/cg.pdb"
        at_path = f"{input_dir_path}/{residue_idx}/at.pdb"


        # Load the files
        parser = PDBParser(QUIET=True)
        cg_structure = parser.get_structure(residue_idx, cg_path)
        at_structure = parser.get_structure(residue_idx, at_path)

        print(cg_structure, at_structure)

        # Get the residues
        cg_residue = list(cg_structure.get_residues())[0]
        at_residue = list(at_structure.get_residues())[0]

        # Get the atoms
        cg_atoms = list(cg_residue.get_atoms())
        at_atoms = list(at_residue.get_atoms())

        # Make a 200x200x200 box for coordinates
        X_MAX = 200
        Y_MAX = 200
        Z_MAX = 200

        # Make the cg data (batchsize, 12, 8)
        for j, bead in enumerate(cg_atoms):
            # Get coordinates
            x, y, z = bead.get_coord()

            # Make the coordinates relative to the box
            X[i, j, 0] = x / X_MAX
            X[i, j, 1] = y / Y_MAX
            X[i, j, 2] = z / Z_MAX

            # Make one hot encoding for the bead type
            X[i, j, 3:8] = 0
            bead_type_id = DOPC_BEAD_TYPE_NAME_IDS[DOPC_CG_NAME_TO_TYPE_MAP[bead.get_name(
            )]]
            X[i, j, 3 + bead_type_id] = 1

        # Make the at data (batchsize, 138, 8)
        for j, atom in enumerate(at_atoms):
            # Get coordinates
            x, y, z = atom.get_coord()

            # Make the coordinates relative to the box
            Y[i, j, 0] = x / X_MAX
            Y[i, j, 1] = y / Y_MAX
            Y[i, j, 2] = z / Z_MAX

            # Make one hot encoding for the bead type
            Y[i, j, 3:8] = 0
            at_type_id = DOPC_ELEMENT_TYPE_NAME_IDS[atom.element]
            Y[i, j, 3 + at_type_id] = 1

    # Convert to tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    return X, Y



def cg_xyz_to_pdb_data(X, output_dir_path):
    """
        Converts a xyz file to a pdb file
    """

    batch_size = X.shape[0]

    # Make the output directory
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Loop over the batch
    for i in range(batch_size):
        # Make a 200x200x200 box for coordinates
        X_MAX = 200
        Y_MAX = 200
        Z_MAX = 200


        # Get the relative coordinates
        coords = X[i, :, :3]

        # Get bead_type index
        bead_type = X[i, :, 3:]
        bead_type_idx = np.argmax(bead_type, axis=1)

        # Make the structure
        residue_id = (' ', i, ' ')
        residue_name = 'DOP'
        residue = Residue.Residue(residue_id, residue_name, segid='')

        # Add the atoms
        for j, coord in enumerate(coords):

            # Recover the original coordinates
            x = float(coord[0] * X_MAX)
            y = float(coord[1] * Y_MAX)
            z = float(coord[2] * Z_MAX)

            # Convert to number with 5 total decimals including the sign and decimals before the point
            # x = "{:2.3f}".format(x)
            # y = "{:2.3f}".format(y)
            # z = "{:2.3f}".format(z)

            # Find the bead type in dict name -> id
            
            bead_name = [
                "NC3",
                "PO4",
                "GL1",
                "GL2",
                "C1A",
                "D2A",
                "C3A",
                "C4A",
                "C1B",
                "D2B",
                "C3B",
                "C4B",
            ][ j ]
            
            name_to_ele = {
                "NC3":  "N",
                "PO4":  "P",
                "GL1":  "X",
                "GL2":  "X",
                "C1A":  "C",
                "D2A":  "D",
                "C3A":  "C",
                "C4A":  "C",
                "C1B":  "C",
                "D2B":  "D",
                "C3B":  "C",
                "C4B":  "C",
            }

            element = name_to_ele[bead_name]

            # Add the atom
            atom = Atom.Atom(
                name=bead_name,
                coord=(x, y, z),
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {bead_name} ",
                serial_number=j,
                element=element
            )
            residue.add(atom)        

        io = PDBIO()
        io.set_structure(residue)
        io.save(f"{output_dir_path}/cg_{i}.pdb", preserve_atom_numbering=True)

def at_xyz_to_pdb_data(X, output_dir_path):
    """
        Converts a xyz file to a pdb file
    """

    batch_size = X.shape[0]

    # Make the output directory
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Loop over the batch
    for i in range(batch_size):
        # Make a 200x200x200 box for coordinates
        X_MAX = 200
        Y_MAX = 200
        Z_MAX = 200


        # Get the relative coordinates
        coords = X[i, :, :3]

        # Get bead_type index
        bead_type = X[i, :, 3:]
        bead_type_idx = np.argmax(bead_type, axis=1)

        # Make the structure
        residue_id = (' ', i, ' ')
        residue_name = 'DOP'
        residue = Residue.Residue(residue_id, residue_name, segid='')

        # Add the atoms
        for j, coord in enumerate(coords):

            # Recover the original coordinates
            x = float(coord[0] * X_MAX)
            y = float(coord[1] * Y_MAX)
            z = float(coord[2] * Z_MAX)

            # Convert to number with 5 total decimals including the sign and decimals before the point
            # x = "{:2.3f}".format(x)
            # y = "{:2.3f}".format(y)
            # z = "{:2.3f}".format(z)

            # Find the bead type in dict name -> id
            
            bead_name = [
                " N  ",
                " C12",
                "H12A",
                "H12B",
                " C13",
                "H13A",
                "H13B",
                "H13C",
                " C14",
                "H14A",
                "H14B",
                "H14C",
                " C15",
                "H15A",
                "H15B",
                "H15C",
                " C11",
                "H11A",
                "H11B",
                " P  ",
                " O13",
                " O14",
                " O12",
                " O11",
                " C1 ",
                " HA ",
                " HB ",
                " C2 ",
                " HS ",
                " O21",
                " C21",
                " O22",
                " C22",
                " H2R",
                " H2S",
                " C3 ",
                " HX ",
                " HY ",
                " O31",
                " C31",
                " O32",
                " C32",
                " H2X",
                " H2Y",
                " C23",
                " H3R",
                " H3S",
                " C24",
                " H4R",
                " H4S",
                " C25",
                " H5R",
                " H5S",
                " C26",
                " H6R",
                " H6S",
                " C27",
                " H7R",
                " H7S",
                " C28",
                " H8R",
                " H8S",
                " C29",
                " H9R",
                "C210",
                "H10R",
                "C211",
                "H11R",
                "H11S",
                "C212",
                "H12R",
                "H12S",
                "C213",
                "H13R",
                "H13S",
                "C214",
                "H14R",
                "H14S",
                "C215",
                "H15R",
                "H15S",
                "C216",
                "H16R",
                "H16S",
                "C217",
                "H17R",
                "H17S",
                "C218",
                "H18R",
                "H18S",
                "H18T",
                " C33",
                " H3X",
                " H3Y",
                " C34",
                " H4X",
                " H4Y",
                " C35",
                " H5X",
                " H5Y",
                " C36",
                " H6X",
                " H6Y",
                " C37",
                " H7X",
                " H7Y",
                " C38",
                " H8X",
                " H8Y",
                " C39",
                " H9X",
                "C310",
                "H10X",
                "C311",
                "H11X",
                "H11Y",
                "C312",
                "H12X",
                "H12Y",
                "C313",
                "H13X",
                "H13Y",
                "C314",
                "H14X",
                "H14Y",
                "C315",
                "H15X",
                "H15Y",
                "C316",
                "H16X",
                "H16Y",
                "C317",
                "H17X",
                "H17Y",
                "C318",
                "H18X",
                "H18Y",
                "H18Z",
            ][ j ]
            
            name_to_ele = {
                " N  ": "N",
                " C12": "C",
                "H12A": "H",
                "H12B": "H",
                " C13": "C",
                "H13A": "H",
                "H13B": "H",
                "H13C": "H",
                " C14": "C",
                "H14A": "H",
                "H14B": "H",
                "H14C": "H",
                " C15": "C",
                "H15A": "H",
                "H15B": "H",
                "H15C": "H",
                " C11": "C",
                "H11A": "H",
                "H11B": "H",
                " P  ": "P",
                " O13": "O",
                " O14": "O",
                " O12": "O",
                " O11": "O",
                " C1 ": "C",
                " HA ": "H",
                " HB ": "H",
                " C2 ": "C",
                " HS ": "H",
                " O21": "O",
                " C21": "C",
                " O22": "O",
                " C22": "C",
                " H2R": "H",
                " H2S": "H",
                " C3 ": "C",
                " HX ": "H",
                " HY ": "H",
                " O31": "O",
                " C31": "C",
                " O32": "O",
                " C32": "C",
                " H2X": "H",
                " H2Y": "H",
                " C23": "C",
                " H3R": "H",
                " H3S": "H",
                " C24": "C",
                " H4R": "H",
                " H4S": "H",
                " C25": "C",
                " H5R": "H",
                " H5S": "H",
                " C26": "C",
                " H6R": "H",
                " H6S": "H",
                " C27": "C",
                " H7R": "H",
                " H7S": "H",
                " C28": "C",
                " H8R": "H",
                " H8S": "H",
                " C29": "C",
                " H9R": "H",
                "C210": "C",
                "H10R": "H",
                "C211": "C",
                "H11R": "H",
                "H11S": "H",
                "C212": "C",
                "H12R": "H",
                "H12S": "H",
                "C213": "C",
                "H13R": "H",
                "H13S": "H",
                "C214": "C",
                "H14R": "H",
                "H14S": "H",
                "C215": "C",
                "H15R": "H",
                "H15S": "H",
                "C216": "C",
                "H16R": "H",
                "H16S": "H",
                "C217": "C",
                "H17R": "H",
                "H17S": "H",
                "C218": "C",
                "H18R": "H",
                "H18S": "H",
                "H18T": "H",
                " C33": "C",
                " H3X": "H",
                " H3Y": "H",
                " C34": "C",
                " H4X": "H",
                " H4Y": "H",
                " C35": "C",
                " H5X": "H",
                " H5Y": "H",
                " C36": "C",
                " H6X": "H",
                " H6Y": "H",
                " C37": "C",
                " H7X": "H",
                " H7Y": "H",
                " C38": "C",
                " H8X": "H",
                " H8Y": "H",
                " C39": "C",
                " H9X": "H",
                "C310": "C",
                "H10X": "H",
                "C311": "C",
                "H11X": "H",
                "H11Y": "H",
                "C312": "C",
                "H12X": "H",
                "H12Y": "H",
                "C313": "C",
                "H13X": "H",
                "H13Y": "H",
                "C314": "C",
                "H14X": "H",
                "H14Y": "H",
                "C315": "C",
                "H15X": "H",
                "H15Y": "H",
                "C316": "C",
                "H16X": "H",
                "H16Y": "H",
                "C317": "C",
                "H17X": "H",
                "H17Y": "H",
                "C318": "C",
                "H18X": "H",
                "H18Y": "H",
                "H18Z": "H",
            }

            element = name_to_ele[bead_name]

            # Add the atom
            atom = Atom.Atom(
                name=bead_name,
                coord=(x, y, z),
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {bead_name} ",
                serial_number=j,
                element=element
            )
            residue.add(atom)

        io = PDBIO()
        io.set_structure(residue)
        io.save(f"{output_dir_path}/at_{i}.pdb", preserve_atom_numbering=True)
