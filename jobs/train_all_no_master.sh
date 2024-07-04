#!/bin/bash
#SBATCH --job-name=MASTER
#SBATCH --output=./jobs/logs/no-master-%j.log
#SBATCH --error=./jobs/logs/no-master-%j.err
#SBATCH --partition=deflt
#SBATCH --nice=0


date
echo "####################### MASTER ###########################"


bond_ics=(
    "1"
    "3"
    "11"
    "16"
    "17"
    "21"
    "22"
    "23"
    "27"
    "31"
    "32"
    "33"
    "34"
    "37"
    "38"
    "41"
    "42"
    "46"
    "47"
    "48"
    "49"
    "52"
)
angle_ics=(
    "53"
    "54"
    "55"
    "56"
    "57"
    "58"
    "59"
    "60"
    "61"
    "62"
    "63"
    "64"
    "65"
    "66"
    "67"
    "68"
    "69"
    "70"
    "71"
    "72"
    "73"
    "74"
    "75"
    "76"
    "77"
    "78"
    "79"
    "81"
    "82"
    "83"
    "84"
    "85"
    "86"
    "87"
    "88"
    "89"
    "90"
    "91"
    "92"
    "93"
    "94"
    "95"
    "96"
    "97"
    "98"
    "99"
    "100"
    "101"
    "102"
    "103"
    "104"
    "105"
    "106"
    "107"
    "108"
    "109"
    "110"
    "111"
    "112"
    "113"
)
dihedral_ics=(
    "114"
    "115"
    "116"
    "117"
    "118"
    "119"
    "120"
    "121"
    "122"
    "123"
    "124"
    "125"
    "126"
    "127"
    "128"
    "129"
    "130"
    "131"
    "132"
    "133"
    "134"
    "135"
    "136"
    "137"
    "138"
    "139"
    "140"
    "141"
    "142"
    "143"
    "144"
    "145"
    "146"
    "147"
    "148"
    "149"
    "150"
    "151"
    "152"
    "153"
    "154"
    "155"
    "156"
    "157"
    "158"
    "159"
    "160"
    "161"
    "162"
    "163"
    "164"
    "165"
    "166"
    "167"
    "168"
    "169"
    "170"
    "171"
    "172"
)

# Read data dir from config.ini
data_dir=$(grep "data_path" config.ini | cut -d "=" -f 2 | cut -d ";" -f 1 | xargs)
hist_dir=$data_dir/hist

# Get all the hist files that contain $1 in their name
hist_files=$(ls $hist_dir | grep prod)

overall_epochs=0
overall_target_epochs=0

echo "Starting to check the training history files for the target epochs"

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

    # Set epoch and val loss 0 if not found
    if [ -z "$epoch" ]; then
        epoch=0
    fi

    if [ -z "$val_loss" ]; then
        val_loss=0
    fi

    # Add one epoch to $epoch
    epoch=$((epoch + 1))

    # If epoch is bigger than target epoch, set it to target epoch
    if [ $epoch -gt $target_epoch ]; then
        epoch=$target_epoch
    fi

    # Remove "train_" from the hist_file
    hist_file=$(echo $hist_file | sed 's/training_history_//g')
    hist_file=$(echo $hist_file | sed 's/.csv//g')

    # Make epoch and target epoch the same length
    epoch=$(printf "%-4s" $epoch)
    target_epoch=$(printf "%-4s" $target_epoch)

    ic_index=$(echo $hist_file | cut -d "_" -f 4)


    # If epoch is equal to target_epoch, we will remove the corresponding IC from the lists above
    if [ $epoch -eq $target_epoch ]; then
        # Remove the IC from the lists by creating a new list without the elemen
        new_bond_ics=()
        new_angle_ics=()
        new_dihedral_ics=()

        for ic in "${bond_ics[@]}"; do
            if [ $ic -ne $ic_index ]; then
                new_bond_ics+=($ic)
            fi
        done

        for ic in "${angle_ics[@]}"; do
            if [ $ic -ne $ic_index ]; then
                new_angle_ics+=($ic)
            fi
        done

        for ic in "${dihedral_ics[@]}"; do
            if [ $ic -ne $ic_index ]; then
                new_dihedral_ics+=($ic)
            fi
        done

        bond_ics=("${new_bond_ics[@]}")
        angle_ics=("${new_angle_ics[@]}")
        dihedral_ics=("${new_dihedral_ics[@]}")
    fi
done


# Shuffle the ICs
bond_ics=($(shuf -e "${bond_ics[@]}"))
angle_ics=($(shuf -e "${angle_ics[@]}"))
dihedral_ics=($(shuf -e "${dihedral_ics[@]}"))

echo "Bond ICs left: ${#bond_ics[@]}"
echo "Angle ICs left: ${#angle_ics[@]}"
echo "Dihedral ICs left: ${#dihedral_ics[@]}"

# Make folder to store logs
mkdir -p ./jobs/logs/$SLURM_JOBID
mkdir -p ./jobs/logs/$SLURM_JOBID/bonds
mkdir -p ./jobs/logs/$SLURM_JOBID/angles
mkdir -p ./jobs/logs/$SLURM_JOBID/dihedrals

blength=${#bond_ics[@]}
alength=${#angle_ics[@]}
dlength=${#dihedral_ics[@]}

z=0

for (( i=0; i<$blength; i+=4 )); do
    package=("${bond_ics[@]:i:4}")
    ic1=${bond_ics[i]}
    
    if (( i+1 < blength )); then
        ic2=${bond_ics[i+1]}
    else
        ic2=""
    fi
    
    if (( i+2 < blength )); then
        ic3=${bond_ics[i+2]}
    else
        ic3=""
    fi
    
    if (( i+3 < blength )); then
        ic4=${bond_ics[i+3]}
    else
        ic4=""
    fi

    z=$((z+1))

    echo "[$z]" Bonds: Starting package $i train for ics: "${package[@]}"
    sbatch --gpus-per-task=1 --gres=gpu:1 --job-name=B$i --exclusive=user --mem=0 --output=./jobs/logs/$SLURM_JOBID/bonds/$i.log --error=./jobs/logs/$SLURM_JOBID/bonds/$i.err --wrap="jobs/train_single_no_purge.sh smaug_bond $ic1 $ic2 $ic3 $ic4"
done

for (( i=0; i<$alength; i+=4 )); do
    package=("${angle_ics[@]:i:4}")
    ic1=${angle_ics[i]}
    
    if (( i+1 < alength )); then
        ic2=${angle_ics[i+1]}
    else
        ic2=""
    fi
    
    if (( i+2 < alength )); then
        ic3=${angle_ics[i+2]}
    else
        ic3=""
    fi
    
    if (( i+3 < alength )); then
        ic4=${angle_ics[i+3]}
    else
        ic4=""
    fi

    z=$((z+1))

    echo "[$z]" Angles: Starting package $i train for ics: "${package[@]}"
    sbatch --gpus-per-task=1 --gres=gpu:1 --job-name=A$i --exclusive=user --mem=0 --output=./jobs/logs/$SLURM_JOBID/angles/$i.log --error=./jobs/logs/$SLURM_JOBID/angles/$i.err --wrap="jobs/train_single_no_purge.sh smaug_angle $ic1 $ic2 $ic3 $ic4"
done

for (( i=0; i<$dlength; i+=4 )); do
    package=("${dihedral_ics[@]:i:4}")
    ic1=${dihedral_ics[i]}
    
    if (( i+1 < dlength )); then
        ic2=${dihedral_ics[i+1]}
    else
        ic2=""
    fi
    
    if (( i+2 < dlength )); then
        ic3=${dihedral_ics[i+2]}
    else
        ic3=""
    fi
    
    if (( i+3 < dlength )); then
        ic4=${dihedral_ics[i+3]}
    else
        ic4=""
    fi

    z=$((z+1))

    echo "[$z]" Dihedrals: Starting package $i train for ics: "${package[@]}"
    sbatch --gpus-per-task=1 --gres=gpu:1 --job-name=D$i --exclusive=user --mem=0 --output=./jobs/logs/$SLURM_JOBID/dihedrals/$i.log --error=./jobs/logs/$SLURM_JOBID/dihedrals/$i.err --wrap="jobs/train_single_no_purge.sh smaug_angle $ic1 $ic2 $ic3 $ic4"
done