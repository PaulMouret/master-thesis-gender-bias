#!/bin/bash

#SBATCH -J create_known_datasets					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o create_known_datasets.out				  # name of output file for this submission script
#SBATCH -e create_known_datasets.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 create_known_datasets.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace