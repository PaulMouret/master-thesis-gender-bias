#!/bin/bash

#SBATCH -J debugging_inference					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o debugging_inference.out				  # name of output file for this submission script
#SBATCH -e debugging_inference.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 debugging_inference.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace