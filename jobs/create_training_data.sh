#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=./logs/jobs/data-gen-%j.log
#SBATCH --error=./logs/jobs/data-gen-%j.err
#SBATCH --nice=0
#SBATCH --nodes=1
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

echo "####################### CODE ###########################"

python src/generate_membrane_data.py

python src/generate_molecule_data_fast.py