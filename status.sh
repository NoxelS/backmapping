#!/bin/bash


# This script checks all hist files that contain $1 in their name for the current epoch and prints the status of the training
# $1: the name of the model

# Check if the user has provided the name of the model
if [ -z "$1" ]; then
    echo "Please provide the name of the model"
    exit 1
fi

# Read data dir from config.ini
data_dir=$(grep "data_path" config.ini | cut -d "=" -f 2 | cut -d ";" -f 1 | xargs)
hist_dir=$data_dir/hist

# Get all the hist files that contain $1 in their name
hist_files=$(ls $hist_dir | grep $1)

# Loop through all the hist files
for hist_file in $hist_files; do
    # Get the current epoch
    epoch=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 1)
    # Get the current loss
    loss=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 2)
    # Get the current accuracy
    accuracy=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 3)
    # Get the current val_loss
    val_loss=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 4)
    # Get the current val_accuracy
    val_accuracy=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 5)
    # Print the status of the training
    echo "Model: $hist_file, Epoch: $epoch, Loss: $loss, Accuracy: $accuracy, Val_loss: $val_loss, Val_accuracy: $val_accuracy"
done