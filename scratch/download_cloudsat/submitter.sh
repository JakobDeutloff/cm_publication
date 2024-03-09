#!/bin/bash
#SBATCH --job-name=cloudsat # Specify job name
#SBATCH --output=cloudsat.o%j # name for standard output log file
#SBATCH --error=cloudsat.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=mh1126
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0

# execute python script in respective environment 
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/download_cloudsat/download.py $1 $2 