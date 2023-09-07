#!/bin/bash
# #SBATCH --nice=0
# #SBATCH --nodes=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gpus=0
# #SBATCH --mem=60000
# #SBATCH --mem-per-gpu=140G            # make sure to at least get 140GB of total vram
# #SBATCH --mem=140G                    # make sure to at least get 140GB of total vram
# #SBTACH --contiguous
#SBTACH --exclusive
#SBTACH --mem=0
# #SBATCH --partition=long
#SBATCH --time=00:03:00:00              # make sure to change the time if running something real
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends
#SBATCH --mail-type=fail                # send email if job fails
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
echo "####################### TRAINING FOR $1 ###########################"

python src/train_one_with_neighborhood.py $1

echo "####################### POST ANALYSIS ###########################"
date
python src/plot_cluster_hist.py
echo "Finished!"