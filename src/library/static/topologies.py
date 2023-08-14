
POINTER_TYPES = ["[moleculetype]", "[atoms]",
                 "[bonds]", "[angles]", "[pairs]"]

DOPC_CG_NAME_TO_TYPE_MAP = {
    'NC3': 'Q0',
    'PO4': 'Qa',
    'GL1': 'Na',
    'GL2': 'Na',
    'C1A': 'C1',
    'D2A': 'C3',
    'C3A': 'C1',
    'C4A': 'C1',
    'C1B': 'C1',
    'D2B': 'C3',
    'C3B': 'C1',
    'C4B': 'C1',
}

DOPC_BEAD_TYPE_NAME_IDS = {
    'Q0': 0,
    'Qa': 1,
    'Na': 2,
    'C1': 3,
    'C3': 4,
}

DOPC_ELEMENT_TYPE_NAME_IDS = {
    'C': 0,
    'N': 1,
    'O': 2,
    'P': 3,
    'H': 4,
}


DOPC_CG_BOND_MAP = {
    1: 2,
    2: 3,
    3: [4, 5],
    4: 9,
    5: 6,
    6: 7,
    7: 8,
    9: 10,
    10: 11,
    11: 12
}

def cg_name_to_type_dict(residue_map_path):
    """
        Returns a dict of CG residue names to CG residue types from a given residue map file
    """
    residue_map = {}
    type_pointer = None

    with open(residue_map_path, 'r') as f:
        for line in f.readlines():
            # Skip comments and empty lines
            if line.startswith(';') or len(line.split()) == 0:
                continue
        
            # Check if the line is a pointer
            if line.replace(" ", "").split()[0] in POINTER_TYPES:
                type_pointer = line.replace(" ", "").split()[0]
                continue

            # Check if the line is an atom type
            if type_pointer == POINTER_TYPES[1]:
                residue_map[line.split()[4]] = line.split()[1]

    return residue_map


def cg_bond_map_dict(residue_map_path):
    """
        Returns a dict of CG residue names to CG residue types from a given residue map file
    """
    bond_map = dict()
    type_pointer = None

    with open(residue_map_path, 'r') as f:
        for line in f.readlines():
            # Skip comments and empty lines
            if line.startswith(';') or len(line.split()) == 0:
                continue

            # Check if the line is a pointer
            if line.replace(" ", "").split()[0] in POINTER_TYPES:
                type_pointer = line.replace(" ", "").split()[0]
                continue

            # Check if the line is a bond type
            if type_pointer == POINTER_TYPES[2]:
                if int(line.split()[0]) not in bond_map.keys():
                    bond_map[int(line.split()[0])] = [ int(line.split()[1]) ]
                else:
                    bond_map[int(line.split()[0])].append(int(line.split()[1]))

    return bond_map