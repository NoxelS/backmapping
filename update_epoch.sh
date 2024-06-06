#!/bin/bash

echo $SLURM_JOB_NAME

epoch=$1
jobname=$SLURM_JOB_NAME
newname="${jobname}_${epoch}"
scontrol update jobid=$SLURM_JOBID jobname=$newname

echo $SLURM_JOB_NAME