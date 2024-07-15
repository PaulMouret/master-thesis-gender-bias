#!/bin/bash

#SBATCH -J creating_dataset					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o creating_dataset.out				  # name of output file for this submission script
#SBATCH -e creating_dataset.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 creating_dataset.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
