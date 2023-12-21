#!/bin/bash
# HINT:
# * The "%j" in the log file names means that the job id will be inserted
#SBATCH --job-name=download cloudsat # Specify job name
#SBATCH --output=cloudsat_download.o%j # name for standard output log file
#SBATCH --error=cloudsat_download.e%j # name for standard error output log
#SBATCH --partition=compute # partition name
#SBATCH --time=08:00:00 # Set a limit on the total run time
#SBATCH --account=bm1183 # Charge resources on this project
#SBATCH --mem=6GB

# Directory to save files in
cd /work/bm1183/m301049/cloudsat

# 2C-ICE
mkdir ice
cd ice 
sftp jakob.deutloffATmpimet.mpg.de@www.cloudsat.cira.colostate.edu:Data/2C-ICE.P1_R05/2014/090/*.hdf

# 2B-FLXHR-LIDAR
mkdir radiation
cd radiation
sftp jakob.deutloffATmpimet.mpg.de@www.cloudsat.cira.colostate.edu:Data/2B-FLXHR-LIDAR.P2_R05/2014/090/*.hdf

# AUX
mkdir aux
cd aux
sftp jakob.deutloffATmpimet.mpg.de@www.cloudsat.cira.colostate.edu:Data/ECMWF-AUX.P1_R05/2014/090/*.hdf
