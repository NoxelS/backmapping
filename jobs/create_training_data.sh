#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=./logs/jobs/training-data-%j.log
#SBATCH --error=./logs/jobs/training-data-%j.err
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

echo "####################### CODE ###########################"
date
python src/generate_data.py
date
# echo "####################### POST ANALYSIS ###########################"
# echo "Finished!"