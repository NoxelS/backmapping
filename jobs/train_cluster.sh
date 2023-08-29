#!/bin/bash
#SBATCH --job-name=AI_HOST
#SBATCH --output=./jobs/logs/host-%j.log
#SBATCH --error=./jobs/logs/host-%j.err
# #SBATCH --nice=0
# #SBATCH --nodes=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gpus=0
# #SBATCH --mem=60000
# #SBATCH --mem-per-gpu=140G        # make sure to at least get 140GB of total vram
# #SBATCH --mem=140G                # make sure to at least get 140GB of total vram
# #SBTACH --contiguous
#SBTACH --exclusive
#SBTACH --mem=0
#SBATCH --partition=long
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
    # "N"   # We don't fit N as this is the reference atom
    "C12"
    "C13"
    "C14"
    "C15"
    "C11"
    "P"
    "O13"
    "O14"
    "O12"
    "O11"
    "C1"
    "C2"
    "O21"
    "C21"
    "O22"
    "C22"
    "C3"
    "O31"
    "C31"
    "O32"
    "C32"
    "C23"
    "C24"
    "C25"
    "C26"
    "C27"
    "C28"
    "C29"
    "C210"
    "C211"
    "C212"
    "C213"
    "C214"
    "C215"
    "C216"
    "C217"
    "C218"
    "C33"
    "C34"
    "C35"
    "C36"
    "C37"
    "C38"
    "C39"
    "C310"
    "C311"
    "C312"
    "C313"
    "C314"
    "C315"
    "C316"
    "C317"
    "C318"
)

nodes_to_use=(
    "fang41"
    "fang42"
    "fang43"
    "fang47"
    "fang48"
    "fang49"
)

length=${#atoms_to_fit[@]}
node_length=${#nodes_to_use[@]}
current_date=$(date +%Y-%m-%d_%H-%M-%S)
folder_name=cluster-$current_date

echo Start training for $length atoms on $node_length nodes

mkdir -p ./jobs/logs/$folder_name

for ((i = 0; i < length; i++))
do
    current_node=${nodes_to_use[i % node_length]}  # Cycle through nodes
    current_atom=${atoms_to_fit[i]}
    sbatch --nodelist=$current_node --job-name=AI_T_$current_atom --exclusive --gres=gpu:1 --output=./jobs/logs/$folder_name/$current_atom.log --error=./jobs/logs/$folder_name/$current_atom.err --wrap="jobs/train_single.sh $current_atom"
    # sbatch --test-only --nodelist=$current_node --job-name=T_$i --mem=0 --exclusive --output=./jobs/logs/master-$SLURM_JOBID/$i.log --error=./jobs/logs/master-$SLURM_JOBID/$i.err --wrap="jobs/train_single.sh $current_node"
done

# Move the hosts log file to the folder
mv ./jobs/logs/host-$SLURM_JOBID.log ./jobs/logs/$folder_name/host-$SLURM_JOBID.log
mv ./jobs/logs/host-$SLURM_JOBID.err ./jobs/logs/$folder_name/host-$SLURM_JOBID.err

# Remove err if empty
if [ ! -s ./jobs/logs/$folder_name/host-$SLURM_JOBID.err ]
then
    rm ./jobs/logs/$folder_name/host-$SLURM_JOBID.err
fi