import os
import pickle

from memdof import ExtendedTopologyInfo, calc_IDOF, parse_PTB, parse_topology

from library.config import Keys, config

# Load paths from the config
PATH_TO_TOPOLOGY = "data/topologies/DOPC.itp"  # Path to the GROMACS topology file (.itp)
PATH_TO_PDB = os.path.join(config(Keys.DATA_PATH), "membranes", "0", "at.pdb")  # Path to the PDB file
TOPOLOGY_PLOT_PATH = os.path.join(config(Keys.DATA_PATH), "ic_dof_analysis")  # Path to the folder where the plots will be saved
CSV_PATH = os.path.join(config(Keys.DATA_PATH), "ic_dof_analysis")  # Path to the folder where the CSV files will be saved


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

    return extended_topology_info


if __name__ == "__main__":
    topo = generate_extended_topology()
    print(topo)
