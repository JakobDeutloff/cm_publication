#!/bin/bash
#SBATCH --job-name=proc_data # Specify job name
#SBATCH --output=proc_data.o%j # name for standard output log file
#SBATCH --error=proc_data.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=mh1126
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/m/m301049/HcModel/"

# execute python script in respective environment 
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/process_data/derive_sample_vars.py
