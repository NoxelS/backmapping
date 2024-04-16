from library.datagen.membranes import generate_membrane_data
from library.datagen.misc import unzip_raw_files
from library.datagen.molecules import generate_molecule_data
from library.datagen.topology import generate_extended_topology

# We use a 10:1 ratio for training and validation data so the batch sizes are optimal
MAX_TRAINING_DATA = 1 * (10 + 1) * 1024

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
    Now we use the membranes to generate the molecule data. This will create a folder with all the molecules in the
    <DATA_PREFIX>/training folder.
"""
generate_molecule_data(MAX_TRAINING_DATA)
