#!/bin/bash

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

overall_epochs=0
overall_target_epochs=0

# Loop through all the hist files
for hist_file in $hist_files; do
    # Get the current epoch
    epoch=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 1 | cut -d "," -f 1)
    # Get the current val_loss
    val_loss=$(tail -n 1 $hist_dir/$hist_file | cut -d " " -f 4 | cut -d "," -f 6)

    # Get the config name out of the hist name
    config_name=$(echo $hist_file | sed 's/training_history_//g')
    config_name=$(echo $config_name | sed 's/.csv//g')
    config_name=$(echo $config_name | sed 's/_[0-9]*$//g')

    # Get the config file that has model_name_prefix=config_name
    conf_file=$(grep -l "model_name_prefix = $config_name" data/configs/*)

    # Read the target epoch from the config
    target_epoch=$(grep "epochs" $conf_file | cut -d "=" -f 2 | xargs)

    # Add to overall epochs
    overall_epochs=$((overall_epochs + epoch))
    overall_target_epochs=$((overall_target_epochs + target_epoch))

    # Set epoch and val loss 0 if not found
    if [ -z "$epoch" ]; then
        epoch=0
    fi

    if [ -z "$val_loss" ]; then
        val_loss=0
    fi

    # If epoch is bigger than target epoch, set it to target epoch
    if [ $epoch -gt $target_epoch ]; then
        epoch=$target_epoch
    fi

    # Calculate percentage of the current epoch if target_epoch is not 0
    if [ $epoch -eq 0 ]; then
        percentage=0
    else
        percentage=$(echo "scale=4; $epoch / $target_epoch * 100" | bc)
    fi

    # Remove "train_" from the hist_file
    hist_file=$(echo $hist_file | sed 's/training_history_//g')
    hist_file=$(echo $hist_file | sed 's/.csv//g')

    # Print a bar ant the end to represent the percentage
    bar=$(seq -s "#" $(echo "($percentage / 2)" | bc) | sed 's/[0-9]//g')


    # Print . for missing epochs
    for i in $(seq 1 $(echo "50 - ($percentage / 2)" | bc)); do
        bar=$bar"."
    done

    # Percentage to 2 decimal places
    percentage=$(printf "%.2f" $percentage)

    # Make epoch and target epoch the same length
    epoch=$(printf "%-4s" $epoch)
    target_epoch=$(printf "%-4s" $target_epoch)
    percentage=$(printf "%-5s" $percentage)

    # If $2 exists, don't print the status of the training
    if [ ! -z "$2" ]; then
        continue
    fi
    # Print the status of the training
    echo -e "$hist_file\tepoch: $epoch/$target_epoch\t($percentage%)\t[$bar]"
done

# Calculate overall percentage
if [ $overall_target_epochs -eq 0 ]; then
    overall_percentage=0
else
    overall_percentage=$(echo "scale=4; $overall_epochs / $overall_target_epochs * 100" | bc)
fi

# Percentage to 2 decimal places
overall_percentage=$(printf "%.2f" $overall_percentage)

# Print the overall status of the training
echo -e "\nOverall\tepoch: $overall_epochs/$overall_target_epochs\t($overall_percentage%)"