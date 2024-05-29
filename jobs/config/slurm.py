from simple_slurm import Slurm

# TODO: Add the necessary parameters to the Slurm object 
slurm = Slurm(
    job_name="MASTER",
    output="./jobs/logs/no-master-%j.log",
    error="./jobs/logs/no-master-%j.err",
    partition="deflt",
    nice=0,
)

