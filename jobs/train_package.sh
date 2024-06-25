#!/bin/bash

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
echo "####################### TRAINING FOR $2, $3, $4, $5 WITH CONFIG $1 ###########################"

srun python -u src/train.py -v --purge --purge-gen-caches --config $1 $2 &

if [ "$3" != "" ]; then
    srun python -u src/train.py -v --purge --purge-gen-caches --config $1 $3 &
fi
if [ "$4" != "" ]; then
    srun python -u src/train.py -v --purge --purge-gen-caches --config $1 $4 &
fi
if [ "$5" != "" ]; then
    srun python -u src/train.py -v --purge --purge-gen-caches --config $1 $5 &
fi

wait