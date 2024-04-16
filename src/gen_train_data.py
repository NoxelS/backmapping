from library.config import Keys, config
from library.datagen.membranes import generate_membrane_data
from library.datagen.misc import unzip_raw_files
from library.datagen.molecules import generate_molecule_data
from library.datagen.topology import generate_extended_topology, generate_output_structure

"""
    First, we unzip the raw data and move it to a dedicated folder
"""
unzip_raw_files()


"""
    Second, we generate the membrane data based on the raw data structure. Then we will have a folder with all the membranes
    that we can use for training in the <DATA_PREFIX>/membranes folder.
"""
generate_membrane_data()


"""
    Use memdof to generate the extended topology of the molecule. This will include all internal coordinates and a flag
    that indicates if the internal coordinate needs to be fitted or not.
"""
generate_extended_topology()


"""
    Use the generated extended topology to generate the output structure of the network. This will create a folder with
    the structure, a few plots to visualize the internal coordinates and a csv file with the internal coordinates with labels.
"""
generate_output_structure()


"""
    Now we use the membranes to generate the molecule data. This will create a folder with all the molecules in the
    <DATA_PREFIX>/training folder.
"""
generate_molecule_data(config(Keys.MAX_TRAINING_DATA))
