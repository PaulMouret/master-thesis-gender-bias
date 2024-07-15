#!/bin/bash

#SBATCH -J is_deterministic					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o is_deterministic.out				  # name of output file for this submission script
#SBATCH -e is_deterministic.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 is_deterministic.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace