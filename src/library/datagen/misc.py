import os
import sys
import subprocess

from library.config import Keys, config

##### CONFIGURATION #####

DATA_PREFIX = config(Keys.DATA_PATH)
RAW_DATA_PATH = "/localdisk/users/noel/DOPC_CG_2_AA_NEW.zip"

def unzip_raw_files():
    ## Find the raw data zip file and unzip it into the right folder
    if os.path.exists(RAW_DATA_PATH):
        subprocess.call(["unzip", RAW_DATA_PATH, "-d", DATA_PREFIX])
        subprocess.call(["mv", f"{DATA_PREFIX}/DOPC_CG_2_AA_NEW/", f"{DATA_PREFIX}/raw/"])
        
        # TODO: Do other stuff that is required for the raw data