import os

MEMBRANE_PATH = "data/membranes"
RAW_PATH = "data/raw"

index = 0

def copy_file(source, destination):
   with open(source, 'rb') as file:
       myFile = file.read()
   with open(destination, 'wb') as file:
       file.write(myFile)

if not os.path.exists(f"{MEMBRANE_PATH}"):
    os.mkdir(f"{MEMBRANE_PATH}")

for entry in os.scandir(RAW_PATH):
    if os.path.isdir(entry):
        print(f"[{index}] Creating membrane out of {entry.name}")
        
        # Create the folder if it does not exist
        if not os.path.exists(f"{MEMBRANE_PATH}/{index}"):
            os.mkdir(f"{MEMBRANE_PATH}/{index}")
        
        # Copy the files over
        # TODO: Here we also could use the checked membranes for ring interference
        copy_file(f"{RAW_PATH}/{entry.name}/FINAL/final_cg2at_de_novo.pdb", f"{MEMBRANE_PATH}/{index}/at.pdb")
        copy_file(f"{RAW_PATH}/{entry.name}/INPUT/CG_INPUT.pdb", f"{MEMBRANE_PATH}/{index}/cg.pdb")
        
        index += 1