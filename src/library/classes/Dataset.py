import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

class Dataset:
    def __init__(self, name, path, type, parent=None):
        self.name = name
        self.path = path
        self.parent = parent
        self.type = type

    def __repr__(self):
        return f'<Dataset {self.name} at {hex(id(self))}>'
    
    def get_structure(self) -> Structure: 
        parser = PDBParser()
        return parser.get_structure(self.name, self.path)
    
    def get_model(self) -> Model:
        return self.get_structure().get_models()[0]

    def get_chains(self) -> list:
        return self.get_structure().get_chains()

    def get_residues(self) -> list:
        return self.get_structure().get_residues()
