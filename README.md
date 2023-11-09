![example workflow](https://github.com/noxels/backmapping/actions/workflows/tests.yml/badge.svg)

# Backmapping

Welcome to the Backmapping project repository! This project focuses on utilizing Convolutional Neural Network (CNN) architecture powered by TensorFlow and Python to perform backmapping on DOPC lipids.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Molecular dynamics simulations often involve complex lipid systems, such as DOPC lipids. Backmapping is a technique that converts an atomistic representation of a lipid to a coarse-grained representation. In this project, we employ a CNN-based AI model implemented in TensorFlow and Python to perform backmapping on DOPC lipids. This can lead to more efficient simulations and analyses while retaining crucial structural information.

## Features

- **CNN Architecture**: We leverage a Convolutional Neural Network architecture to learn the mapping from atomistic to coarse-grained representation.
- **TensorFlow**: The project is built using the TensorFlow framework, ensuring scalability and efficient computation.
- **Python**: All code is written in Python, making it accessible and easily modifiable.
- **Easy Integration**: The developed model can be seamlessly integrated into existing molecular dynamics workflows.
- **Documentation**: Clear and concise documentation is provided to help users understand and use the backmapping tool effectively.

## Installation

To install the backmapping tool, you can clone the repository and install the dependencies using the following commands:

```bash
git clone
cd backmapping
pip install -r requirements.txt
```

## Usage

### Generating membrane data with generate_membrane_data.py

#### Overview:

This script, `generate_membrane_data.py`, is designed to generate membrane data from a specified raw data directory. It organizes and copies relevant files into a structured membrane directory.

#### File Structure:

The script creates a new directory for each membrane, identified by an index, inside the specified `MEMBRANE_PATH`. Inside each membrane directory, two files are copied:

1. `at.pdb`: This file is sourced from the `FINAL` directory within the raw data, specifically from `final_cg2at_de_novo.pdb`. It plays a crucial role in the membrane generation process.

2. `cg.pdb`: This file is copied from the `INPUT` directory in the raw data, specifically from `CG_INPUT.pdb`. It complements the membrane generation process by providing additional context.

#### Usage:

1. Ensure that the necessary configurations are set in the `config.py` file, particularly the `DATA_PATH`.

2. Run the script. It will scan the specified `RAW_PATH` for directories, create corresponding membrane directories, and copy the required files.

3. The membrane data is now organized in the `MEMBRANE_PATH` directory, ready for further processing.

#### Additional Notes:

- The script prints progress messages, indicating the creation of each membrane and the files being copied.

- There's a placeholder comment suggesting the potential use of checked membranes for ring interference. This is marked as a TODO for potential future enhancements.

- Make sure that the required dependencies are installed, and the file paths in the script match the actual structure of your raw data.


### Generating training data with generate_molecule_data_fast.py

#### Overview:

The script, `generate_molecule_data_fast.py`, efficiently generates molecular data from membrane datasets. It extracts relevant information, organizes it into structured directories, and logs important details.

#### File Paths:

- `input_dir_path`: Path to the directory containing membrane datasets.
- `output_dir_path`: Path to the directory where molecular data will be saved.
- `output_box_table_path`: Path where box sizes are saved.
- `training_dir_path`: Path to the training data directory.

#### Parameters:

- `TRAINING_DATA_MODE`: Boolean flag indicating whether the script is in training data generation mode.
- `MAX_SAMPLES`: Maximum number of samples to generate.
- `NEIGHBOUR_CUTOFF_RADIUS`: Radius for considering neighbors.
- `TARGET_NEIGHBOUR_COUNT`: Target number of neighbors for each residue.

#### Execution:

1. Ensure that the configurations in `config.py` are set, especially `DATA_PATH`.

2. Run the script. It processes membrane datasets, creates molecular data, and organizes it in the specified directories.

3. Check the output:
   - Molecular data is saved in `output_dir_path`.
   - Box sizes are saved in `output_box_table_path`.
   - Training data is saved in `training_dir_path` if in training mode.

4. View statistics:
   - The script prints progress and statistics, such as mean and std of neighbors count.
   - Histograms are plotted to visualize the distribution of neighbors counts.

5. Additional Outputs:
   - `box_sizes_cg.csv` and `box_sizes_at.csv`: Box sizes of CG and AT datasets.
   - `neighborhoods.csv`: List of neighbors for each residue.
   - `mean_distances.csv`: Mean distances to the N atom for each residue.

#### Note:

- The script efficiently handles large datasets without loading them entirely into memory.
- It calculates and logs crucial statistics, aiding in the analysis of generated data.
- Histograms provide insights into the distribution of neighbors counts.


### Running the master resolver wiht master.py

#### Overview:

The `master.py` script acts as the master controller for training multiple models for each atom. It communicates with child processes, manages training progress, and sends updates via email.

#### Configuration:

- `DATA_PREFIX`: Path to the data directory.
- `ENCODING_STOP_SYMBOL`: Symbol used for encoding data.
- `ENCODING_FINISHED_SYMBOL`: Symbol indicating a finished model during decoding.
- `ENCODING_START_SYMBOL`: Symbol indicating a model starting during decoding.
- `EMAIL_SERVER`: SMTP server for sending emails.
- `EMAIL_TARGET`: Email address to receive training history plots.
- `EMAIL_USER`: User sending the email.
- `PORT`: Port for socket communication.

#### Master Socket:

- The master socket communicates with child processes after each epoch.
- Child processes send epoch and validation loss to the master process.

###### Execution:

1. Run the script with the number of models as an argument.

2. The script listens for child processes and manages training progress.

3. Emails are sent:
   - A start email is sent when training begins.
   - Update emails are sent with training history plots after each model finishes.

4. The master socket is closed when all models finish training.

#### Note:

- The script efficiently manages multiple training processes using socket communication.
- Email updates provide insights into the training progress and completion.


### Training a single-atom model with train_one_with_neighborhood.py

#### Overview:

The `train_one_with_neighborhood.py` script is designed for training a model specific to a single atom with neighborhood information. It uses TensorFlow and communicates training progress to the master process through sockets.

#### Configuration:

- `DATA_PREFIX`: Path to the data directory.
- `BATCH_SIZE`: Batch size for training.
- `VALIDATION_SPLIT`: Percentage of data to use for validation.
- `NEIGHBORHOOD_SIZE`: Size of the neighborhood for training.
- `EPOCHS`: Number of training epochs.
- `MODEL_NAME_PREFIX`: Prefix for saving and loading the model.
- `DATA_USAGE`: Data usage strategy during training.
- `USE_TENSORBOARD`: Whether to use TensorBoard for visualization.

#### Network Configuration:

- `cg_size`: Size of the CG input.
- `at_size`: Size of the AT output.
- `strategy`: TensorFlow distribution strategy for synchronization.
- `atom_names_to_fit`: Atom names used for training.
- `atom_name_to_fit`: Atom name to fit during training.

#### Socket Configuration:

- `use_socket`: Flag for using socket communication.
- `host_ip_address`: IP address of the master process.
- `client`: Socket client for communication.

#### Training Configuration:

- `sample_gen`: Generator for a single sample used for testing.
- `train_gen`: Generator for training data.
- `validation_gen`: Generator for validation data.

#### Model Configuration:

- `cnn`: CNN model for training.
- `BackmappingAbsolutePositionLoss`: Custom loss function.
- `test_sample`: Sample used for testing the model.

#### Training Execution:

1. Provide the atom name to fit as a command-line argument.
2. Optionally, provide the master process IP address as a second argument for socket communication.
3. The script prepares data generators and sets up TensorFlow distribution strategy.
4. The model is trained with the specified configuration.
5. Training progress is communicated to the master process through sockets.
6. The model is saved after training.

#### Note:

- The script efficiently trains a model for a specific atom with neighborhood information.
- Socket communication allows coordination with the master process.


### Plot training cluster history with plot_cluster_hist.py

#### Overview:

The `plot_cluster_hist.py` script is responsible for plotting the training history of models. It visualizes Mean Squared Error (MSE) loss over epochs for different atoms, using color-coding based on normalized mean distances.

#### Configuration:

- `PATH_TO_HIST`: Path to the directory containing training history files.

#### Plotting:

- The script reads mean distances from a CSV file.
- For each training history file in the specified directory:
  - Extract the atom name and mean distance.
  - Load the training history CSV.
  - Normalize the mean distances for color-coding.
  - Plot the MSE loss over epochs, color-coded by mean distance.

#### Plot Adjustments:

- Minimum loss is calculated and used for reference in the plot.
- The plot is set to log scale on the y-axis.
- Y-ticks and labels are added for better readability.
- A horizontal line is added at the minimum loss point.
- A label indicating the minimum loss is placed on the plot.

#### Legend:

- Legend with atom names is placed outside the plot.
- A separate legend explains color-coding based on normalized mean distances.

#### Output:

- The final plot is saved as "training_history.png".

#### Execution:

- When executed as the main script, it generates the training history plot.

#### Note:

- The script provides a comprehensive visualization of the training history for different atoms.
- Color-coding based on mean distances enhances the interpretability of the plot.