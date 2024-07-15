#!/bin/bash

#SBATCH -J update_context_dataset_with_manual					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o update_context_dataset_with_manual.out				  # name of output file for this submission script
#SBATCH -e update_context_dataset_with_manual.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 update_context_dataset_with_manual.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
