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

# Make folder to store logs
mkdir -p ./jobs/logs/lr_run_$SLURM_JOBID

for i in {12..25}
do
    sbatch --exclude=fang1,fang8,fang31,fang40,fang54,fang48,fang51,fang52,fang53,fang54 --job-name=LR$i --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/lr_run_$SLURM_JOBID/$i.log --error=./jobs/logs/lr_run_$SLURM_JOBID/$i.err --wrap="jobs/train_single.sh 94 lr_run_$i"
done