import csv
import os
import pickle

from memdof import ExtendedTopologyInfo, calc_IDOF, parse_PTB, parse_topology

from library.config import Keys, config

# Load paths from the config
PATH_TO_TOPOLOGY = "data/topologies/DOPC.itp"  # Path to the GROMACS topology file (.itp)
PATH_TO_PDB = os.path.join(config(Keys.DATA_PATH), "membranes", "0", "at.pdb")  # Path to the PDB file
TOPOLOGY_PLOT_PATH = os.path.join(config(Keys.DATA_PATH), "ic_dof_analysis")  # Path to the folder where the plots will be saved
CSV_PATH = os.path.join(config(Keys.DATA_PATH), "ic_dof_analysis")  # Path to the folder where the CSV files will be saved
NETWORK_STRUCTURE_PATH = os.path.join(config(Keys.DATA_PATH), "network_structure")  # Path to the folder where the CSV files will be saved


def generate_extended_topology() -> ExtendedTopologyInfo:
    """
    Generate the extended topology of the molecule. This will include all internal coordinates and a flag that indicates
    if the internal coordinate needs to be fitted or not.
    """

    # Create dirs if not exist
    os.makedirs(TOPOLOGY_PLOT_PATH) if not os.path.exists(TOPOLOGY_PLOT_PATH) else None
    os.makedirs(CSV_PATH) if not os.path.exists(CSV_PATH) else None

    # Parse the topology information from a GROMACS topology file (.itp)
    topology_info = parse_topology(PATH_TO_TOPOLOGY, ignore_hydrogen=True)

    # Parse the PDB file to extract the structure
    structure, pbc = parse_PTB(PATH_TO_PDB)

    # Calculate the internal degrees of freedom
    extended_topology_info: ExtendedTopologyInfo = calc_IDOF(
        structure, pbc, topology_info, quiet=False, create_plots=True, plots_path=TOPOLOGY_PLOT_PATH, create_csv=True, csv_path=CSV_PATH
    )

    # Save the extended topology information
    with open(os.path.join(TOPOLOGY_PLOT_PATH, "extended_topology.pkl"), "wb") as fp:
        pickle.dump(extended_topology_info, fp)

    # Save the extended topology information as json
    with open(os.path.join(TOPOLOGY_PLOT_PATH, "extended_topology.json"), "w") as fp:
        fp.write(extended_topology_info.json())

    return extended_topology_info


def generate_output_structure(extended_topology: ExtendedTopologyInfo = None):
    """
    Use the generated extended topology to generate the output structure of the network. This will create a folder with
    the structure, a few plots to visualize the internal coordinates and a csv file with the internal coordinates with labels.
    """

    # Load the extended topology information if not provided
    if extended_topology is None:
        with open(os.path.join(TOPOLOGY_PLOT_PATH, "extended_topology.pkl"), "rb") as fp:
            extended_topology = pickle.load(fp)

    # Create dirs if not exist
    if not os.path.exists(NETWORK_STRUCTURE_PATH):
        os.makedirs(NETWORK_STRUCTURE_PATH)

    # Go through all atoms and store them in a csv file
    with open(os.path.join(NETWORK_STRUCTURE_PATH, "atoms.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", dialect="excel", lineterminator="\n")
        writer.writerow(extended_topology.atoms[0].keys())
        [writer.writerow(atom.values()) for atom in extended_topology.atoms]

    # Go through all atoms except the hydrogen atoms and store them in a csv file
    with open(os.path.join(NETWORK_STRUCTURE_PATH, "atoms_no_hydrogen.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", dialect="excel", lineterminator="\n")
        writer.writerow(extended_topology.atoms[0].keys())
        [writer.writerow(atom.values()) for atom in extended_topology.atoms if not atom["atom"].startswith("H")]

    # Go through all degrees of freedom, label them and set fixed flag
    with open(os.path.join(NETWORK_STRUCTURE_PATH, "internal_coordinates.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", dialect="excel", lineterminator="\n")
        writer.writerow(["nr", "type", "label", "label_h", "fixed"])
        nr = 0

        for bond in extended_topology.bonds:
            writer.writerow(
                [
                    nr,
                    "bond",
                    f"{bond['i']}-{bond['j']}",
                    ic_to_hlabel(bond, extended_topology),
                    bond["fixed"],
                ]
            )
            nr += 1

        for angle in extended_topology.angles:
            writer.writerow(
                [
                    nr,
                    "angle",
                    f"{angle['i']}-{angle['j']}-{angle['k']}",
                    ic_to_hlabel(angle, extended_topology),
                    angle["fixed"],
                ]
            )
            nr += 1

        for dihedral in extended_topology.dihedrals:
            writer.writerow(
                [
                    nr,
                    "dihedral",
                    f"{dihedral['i']}-{dihedral['j']}-{dihedral['k']}-{dihedral['l']}",
                    ic_to_hlabel(dihedral, extended_topology),
                    dihedral["fixed"],
                ]
            )
            nr += 1


def get_max_ic_index() -> int:
    """
    Returns the maximum index of the internal coordinates.
    """
    # Read the csv file
    with open(os.path.join(NETWORK_STRUCTURE_PATH, "internal_coordinates.csv"), "r") as f:
        reader = csv.reader(f, delimiter=";")
        # Skip the header
        next(reader)
        # Get the last row
        last_row = list(reader)[-1]
        return int(last_row[0])


def load_extended_topology_info() -> ExtendedTopologyInfo:
    """
    Load the extended topology information from the pickle file.
    """
    with open(os.path.join(TOPOLOGY_PLOT_PATH, "extended_topology.pkl"), "rb") as fp:
        return pickle.load(fp)


def get_ic_from_index(index: int, extended_topology: ExtendedTopologyInfo = None) -> dict:
    if extended_topology is None:
        extended_topology = load_extended_topology_info()

    if index < len(extended_topology.bonds):
        return extended_topology.bonds[index]

    index -= len(extended_topology.bonds)
    if index < len(extended_topology.angles):
        return extended_topology.angles[index]

    index -= len(extended_topology.angles)
    if index < len(extended_topology.dihedrals):
        return extended_topology.dihedrals[index]

    raise ValueError("IC index out of bounds")


def get_ic_type_from_index(index: int, extended_topology: ExtendedTopologyInfo = None) -> str:
    if extended_topology is None:
        extended_topology = load_extended_topology_info()

    if index < len(extended_topology.bonds):
        return "bond"

    index -= len(extended_topology.bonds)
    if index < len(extended_topology.angles):
        return "angle"

    index -= len(extended_topology.angles)
    if index < len(extended_topology.dihedrals):
        return "dihedral"

    raise ValueError("IC index out of bounds")


def get_ic_type(ic: dict, extended_topology: ExtendedTopologyInfo = None) -> str:
    if extended_topology is None:
        extended_topology = load_extended_topology_info()

    if ic in extended_topology.bonds:
        return "bond"

    if ic in extended_topology.angles:
        return "angle"

    if ic in extended_topology.dihedrals:
        return "dihedral"

    raise ValueError(f"IC {ic} not found in extended topology")


def ic_to_hlabel(ic: dict, extended_topology: ExtendedTopologyInfo = None) -> str:
    """
    Return the human readable label of the internal coordinate.
    """

    if extended_topology is None:
        extended_topology = load_extended_topology_info()

    if ic in extended_topology.bonds:
        return f"b{'x' if bool(ic['fixed']) else ''}{extended_topology.atoms[int(ic['i'])]['atom']}-{extended_topology.atoms[int(ic['j'])]['atom']}"

    if ic in extended_topology.angles:
        return f"a{'x' if bool(ic['fixed']) else ''}{extended_topology.atoms[int(ic['i'])]['atom']}-{extended_topology.atoms[int(ic['j'])]['atom']}-{extended_topology.atoms[int(ic['k'])]['atom']}"

    if ic in extended_topology.dihedrals:
        return f"d{'x' if bool(ic['fixed']) else ''}{extended_topology.atoms[int(ic['i'])]['atom']}-{extended_topology.atoms[int(ic['j'])]['atom']}-{extended_topology.atoms[int(ic['k'])]['atom']}-{extended_topology.atoms[int(ic['l'])]['atom']}"

    raise ValueError("IC not found in extended topology")
