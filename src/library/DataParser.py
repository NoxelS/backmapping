import os
from library.classes.Dataset import Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser

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
                os.makedirs(f"data/training/{idx}")

            io.set_structure(cg_dataset.get_structure())
            io.save(f"data/training/{idx}/cg.pdb",
                    ResidueSelector(cg_residue._id[1]), preserve_atom_numbering=True)

            io.set_structure(at_dataset.get_structure())
            io.save(f"data/training/{idx}/at.pdb",
                    ResidueSelector(at_residue._id[1]), preserve_atom_numbering=True)

            idx += 1
