from library.analysis.plots import plot_hist_single

import os
from library.config import config, Keys 

# Find all hist files
hist_files = [file for file in os.listdir(os.path.join(config(Keys.DATA_PATH), "hist")) if ".csv" in file]
indices = [int(file.split("_")[-1].split(".")[0]) for file in hist_files]

for index in indices:
    try:
        plot_hist_single(index)
    except Exception as e:
        print(f"Could not plot for index {index}")
        print(e)