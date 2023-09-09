#!/bin/bash
#SBATCH --job-name=MASTER
#SBATCH --output=./jobs/logs/master-%j.log
#SBATCH --error=./jobs/logs/master-%j.err
#SBATCH --partition=long

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
host_ip_address=$(hostname -i | cut -d' ' -f1)

echo Start training for $length atoms on $node_length nodes

mkdir -p ./jobs/logs/$folder_name

for ((i = 0; i < length; i++))
do
    current_node=${nodes_to_use[i % node_length]}  # Cycle through nodes
    current_atom=${atoms_to_fit[i]}
    sbatch --nodelist=$current_node --job-name=$current_atom --exclusive --gres=gpu:1 --output=./jobs/logs/$folder_name/$current_atom.log --error=./jobs/logs/$folder_name/$current_atom.err --wrap="jobs/train_single.sh $current_atom $host_ip_address"
done

# Start master socket (automatically waits for all jobs to finish)
python src/master.py $length

# Move the hosts log file to the folder
mv ./jobs/logs/master-$SLURM_JOBID.log ./jobs/logs/$folder_name/host-$SLURM_JOBID.log
mv ./jobs/logs/master-$SLURM_JOBID.err ./jobs/logs/$folder_name/host-$SLURM_JOBID.err

# Remove err if empty
if [ ! -s ./jobs/logs/$folder_name/host-$SLURM_JOBID.err ]
then
    rm ./jobs/logs/$folder_name/master-$SLURM_JOBID.err
fi