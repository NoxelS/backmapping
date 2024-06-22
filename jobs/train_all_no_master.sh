#!/bin/bash
#SBATCH --job-name=MASTER
#SBATCH --output=./jobs/logs/no-master-%j.log
#SBATCH --error=./jobs/logs/no-master-%j.err
#SBATCH --partition=deflt
#SBATCH --nice=0

# This block is echoing some SLURM variables
echo "###################### JOB DETAILS ########################"
echo "Job execution start: $(date)"
echo "JobID = $SLURM_JOBID"
echo "Host = $SLURM_JOB_NODELIST"
echo "Jobname = $SLURM_JOB_NAME"
echo "Subcwd = $SLURM_SUBMIT_DIR"
echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "SLURM_GPUS_PER_TASK = $SLURM_GPUS_PER_TASK"
echo "SLURM_GPUS_ON_NODE = $SLURM_GPUS_ON_NODE"
echo ""

date
echo "####################### CODE ###########################"


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
    "53"
)
angle_ics=(
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

# Shuffle the ICs
bond_ics=($(shuf -e "${bond_ics[@]}"))
angle_ics=($(shuf -e "${angle_ics[@]}"))
dihedral_ics=($(shuf -e "${dihedral_ics[@]}"))

# Make folder to store logs
mkdir -p ./jobs/logs/$SLURM_JOBID
mkdir -p ./jobs/logs/$SLURM_JOBID/bonds
mkdir -p ./jobs/logs/$SLURM_JOBID/angles
mkdir -p ./jobs/logs/$SLURM_JOBID/dihedrals

# for ic in "${bond_ics[@]}"
# do
#     sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=B$ic --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/$SLURM_JOBID/bonds/$ic.log --error=./jobs/logs/$SLURM_JOBID/bonds/$ic.err --wrap="jobs/train_single_no_purge.sh $ic smaug_bond"
# done

# for ic in "${angle_ics[@]}"
# do
#     sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=A$ic --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/$SLURM_JOBID/angles/$ic.log --error=./jobs/logs/$SLURM_JOBID/angles/$ic.err --wrap="jobs/train_single_no_purge.sh $ic smaug_angle"
# done

# for ic in "${dihedral_ics[@]}"
# do
#     sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=D$ic --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/$SLURM_JOBID/dihedrals/$ic.log --error=./jobs/logs/$SLURM_JOBID/dihedrals/$ic.err --wrap="jobs/train_single_no_purge.sh $ic smaug_angle"
# done


blength=${#bond_ics[@]}
alength=${#angle_ics[@]}
dlength=${#dihedral_ics[@]}

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

    echo Bonds: Starting package $i train for ics: "${package[@]}"
    sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=PB$i --exclusive --mem=0 --output=./jobs/logs/$SLURM_JOBID/bonds/$i.log --error=./jobs/logs/$SLURM_JOBID/bonds/$i.err --wrap="jobs/train_single_no_purge.sh smaug_bond $ic1 $ic2 $ic3 $ic4"
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


    echo Angles: Starting package $i train for ics: "${package[@]}"
    sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=PA$i --exclusive --mem=0 --output=./jobs/logs/$SLURM_JOBID/angles/$i.log --error=./jobs/logs/$SLURM_JOBID/angles/$i.err --wrap="jobs/train_single_no_purge.sh smaug_angle $ic1 $ic2 $ic3 $ic4"
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

    echo Dihedrals: Starting package $i train for ics: "${package[@]}"
    sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=PD$i --exclusive --mem=0 --output=./jobs/logs/$SLURM_JOBID/dihedrals/$i.log --error=./jobs/logs/$SLURM_JOBID/dihedrals/$i.err --wrap="jobs/train_single_no_purge.sh smaug_angle $ic1 $ic2 $ic3 $ic4"
done