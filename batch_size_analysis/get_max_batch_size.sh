#!/bin/bash

#SBATCH -J get_max_batch_size					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o get_max_batch_size.out				  # name of output file for this submission script
#SBATCH -e get_max_batch_size.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 get_max_batch_size.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace