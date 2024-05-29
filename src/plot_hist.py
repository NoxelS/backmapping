import os

from library.analysis.plots import plot_hist_single
from library.config import Keys, config

# Find all hist files
hist_files = [file for file in os.listdir(os.path.join(config(Keys.DATA_PATH), "hist")) if ".csv" in file]

for file in hist_files:
    try:
        plot_hist_single(file)
    except Exception as e:
        print(f"Could not plot for index {file}")
        print(e)
