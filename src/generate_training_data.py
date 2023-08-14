import os
from Bio.PDB.PDBIO import PDBIO
import time
from datetime import datetime
from library.parser import get_cg_at_datasets

def copy_file(source, destination):
    with open(source, 'rb') as file:
        myFile = file.read()
    with open(destination, 'wb') as file:
        file.write(myFile)

input_dir_path = "data/molecules"
output_dir_path = "data/training"

# Create output dir if it does not exist
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# Creatr /input and /output dirs if they do not exist
if not os.path.exists(f"{output_dir_path}/input"):
    os.makedirs(f"{output_dir_path}/input")

if not os.path.exists(f"{output_dir_path}/output"):
    os.makedirs(f"{output_dir_path}/output")
    
# Loop through all molecules and copy data into flat structures 
for i, molecule in enumerate(os.listdir(input_dir_path)):
    copy_file(f"{input_dir_path}/{molecule}/cg.pdb", f"{output_dir_path}/input/{i}.pdb")
    copy_file(f"{input_dir_path}/{molecule}/at.pdb", f"{output_dir_path}/output/{i}.pdb")
    
    # Print progress
    if i % 1000 == 0:
        timestamp = datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{timestamp}] Copied {i} training samples")
