#!/bin/bash
#SBATCH --job-name=v1_train
#SBATCH --output=./jobs/logs/train-O31-6N-%j.log
#SBATCH --error=./jobs/logs/train-O31-6N-%j.err
#SBATCH --nice=0
#SBATCH --nodes=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gpus=0
# #SBATCH --mem=60000
# #SBATCH --mem-per-gpu=140G        # make sure to at least get 140GB of total vram
# #SBATCH --mem=140G                # make sure to at least get 140GB of total vram
# #SBTACH --contiguous
#SBATCH --partition=deflt
# #SBATCH --time=02:00:00:00             # make sure to change the time if running something real
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=noel@lust.uni-sb.de


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

# Set envs
# export PYENV_ROOT="/home/global/pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"

date
echo "####################### CODE ###########################"

## declare an array variable
atoms_to_fit=(
    "N",
    "C12",
    "C13",
    "C14",
    "C15",
    "C11",
    "P",
    "O13",
    "O14",
    "O12",
    "O11",
    "C1",
    "C2",
    "O21",
    "C21",
    "O22",
    "C22",
    "C3",
    "O31",
    "C31",
    "O32",
    "C32",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C210",
    "C211",
    "C212",
    "C213",
    "C214",
    "C215",
    "C216",
    "C217",
    "C218",
    "C33",
    "C34",
    "C35",
    "C36",
    "C37",
    "C38",
    "C39",
    "C310",
    "C311",
    "C312",
    "C313",
    "C314",
    "C315",
    "C316",
    "C317",
    "C318",
)

length=${#atoms_to_fit[@]}

## now loop through the above array
for i in "${atoms_to_fit[@]}"
do
    echo "$i"
    echo $length
    # python src/train_one_with_neighborhood.py $i
done
   


echo "####################### POST ANALYSIS ###########################"
date
echo "Finished!"