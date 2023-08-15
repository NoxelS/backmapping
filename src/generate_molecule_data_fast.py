import os
from Bio.PDB.PDBIO import PDBIO
import time
from datetime import datetime
from library.parser import get_cg_at_datasets

input_dir_path = "data/membranes"
output_dir_path = "data/molecules"
training_dir_path = "data/training"
TRAINING_DATA_MODE = True
MAX_SAMPLES = (10 + 1) * 1024  # 102400

cg_datasets, at_datasets = [], []
io = PDBIO()
idx_at = 0
idx_cg = 0
current_time = time.time()

def elapsed_time():
    elapsed_time = time.time() - current_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

# Get all CG and AT datasets
cg_datasets, at_datasets = get_cg_at_datasets(
    input_dir_path, CG_PATTERN='cg.pdb', AT_PATTERN='at.pdb')

# Create output dir if it does not exist
if not os.path.exists(output_dir_path) and not TRAINING_DATA_MODE:
    os.makedirs(output_dir_path)
    
if TRAINING_DATA_MODE and not os.path.exists(training_dir_path):
    os.makedirs(training_dir_path)
    os.makedirs(f"{training_dir_path}/input")
    os.makedirs(f"{training_dir_path}/output")

# Loop over both at the same time (these are generators, so they are not loaded into memory immediately)
for i, (cg_dataset, at_dataset) in enumerate(zip(cg_datasets, at_datasets)):
    # Open the CG and AT files
    cg_file = open(cg_dataset.path)
    at_file = open(at_dataset.path)
    
    # Read the CG and AT files
    cg_lines = cg_file.readlines()
    at_lines = at_file.readlines()
    
    # Close the CG and AT files
    cg_file.close()
    at_file.close()
    
    # Remove all lines that are not ATOM lines
    cg_lines = [line for line in cg_lines if line.startswith("ATOM")]
    at_lines = [line for line in at_lines if line.startswith("ATOM")]
    
    last_residue_idx_cg = 0
    last_residue_idx_at = 0
    
    line_stack_cg = []
    line_stack_at = []
    
    # Loop over the lines of the CG files
    for j, cg_line in enumerate(cg_lines):
        # Check if we have reached the max number of samples
        if idx_cg > MAX_SAMPLES:
            break
        
        # Split the lines and get the residue index
        cg_line_split = cg_line.split()
        residue_idx_cg = int(cg_line_split[4])

        # Check if the residue changes or last line
        if residue_idx_cg != last_residue_idx_cg or j == len(cg_lines) - 1:
            # Add the line to the stack
            if j == len(cg_lines) - 1:
                line_stack_cg.append(cg_line)
                        
            # Make folder for the residue if not in training data mode
            if not TRAINING_DATA_MODE:
                if not os.path.exists(f"{output_dir_path}/{i}_{last_residue_idx_cg}"):
                    os.makedirs(f"{output_dir_path}/{i}_{last_residue_idx_cg}")

            # Add TER line to the stack
            line_stack_cg.append(f"TER      13      DOPC    {last_residue_idx_cg}\n")  # <- I don't know why this is needed, but biopython does it too
            line_stack_cg.append("END   ")

            # Set path
            save_path = f"{output_dir_path}/{i}_{last_residue_idx_cg}/cg.pdb" if not TRAINING_DATA_MODE else f"{training_dir_path}/input/{idx_cg}.pdb"
            
            # Save current stack to file
            with open(save_path, "w") as cg_file:
                for line in line_stack_cg:
                    cg_file.write(line)
                    

            line_stack_cg = []
            last_residue_idx_cg = residue_idx_cg
        
            idx_cg += 1

        # Add the line to the stack
        line_stack_cg.append(cg_line)
        
    # Loop over the lines of the AT files
    for j, at_line in enumerate(at_lines):
        # Check if we have reached the max number of samples
        if idx_at > MAX_SAMPLES:
            break
        
        # Split the lines and get the residue index
        at_line_split = at_line.split()
        residue_idx_at = int(at_line_split[4]) - 1 # <- ATOM lines start at 1, CG lines start at 0. I don't know why

        # Check if the residue changes or last line
        if residue_idx_at != last_residue_idx_at or j == len(at_lines) - 1:            
            # Add the line to the stack
            if j == len(at_lines) - 1:
                line_stack_at.append(at_line)
                
            # Make folder for the residue if not in training data mode
            if not TRAINING_DATA_MODE:
                if not os.path.exists(f"{output_dir_path}/{i}_{last_residue_idx_at}"):
                    os.makedirs(f"{output_dir_path}/{i}_{last_residue_idx_at}")            

            # Add TER line to the stack
            line_stack_at.append(f"TER     {(139 + 138*last_residue_idx_at) % 100000}      DOPC    {last_residue_idx_at + 1}\n") # <- I don't know why this is needed, but biopython does it too
            line_stack_at.append("END   ")

            # Save also to training data
            save_path = f"{output_dir_path}/{i}_{last_residue_idx_at}/at.pdb" if not TRAINING_DATA_MODE else f"{training_dir_path}/output/{idx_at}.pdb"

            # Save current stack to file
            with open(save_path, "w") as at_file:
                for line in line_stack_at:
                    at_file.write(line)
                    

            line_stack_at = []
            last_residue_idx_at = residue_idx_at
            
            # Print progress
            if idx_at % 100 == 0:
                timestamp = datetime.fromtimestamp(
                    time.time()).strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"[{timestamp}] Generated {idx_at} training samples ({elapsed_time()})")

            idx_at += 1

        # Add the line to the stack
        line_stack_at.append(at_line)

    if idx_cg > MAX_SAMPLES and idx_at > MAX_SAMPLES:
        break