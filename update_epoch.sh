#!/bin/bash

echo $SLURM_JOB_NAME

epoch=$1
jobname=$SLURM_JOB_NAME
newname="${jobname}_${epoch}"
scontrol update JobId=$SLURM_JOBID JobName=$newname

echo $SLURM_JOB_NAME