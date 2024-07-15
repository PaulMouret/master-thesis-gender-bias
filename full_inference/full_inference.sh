#!/bin/bash

#SBATCH -J full_inference					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o full_inference.out				  # name of output file for this submission script
#SBATCH -e full_inference.err				  # name of error file for this submission script
#SBATCH -G nvidia_geforce_rtx_3090:1

#specific "capacity" commands
#SBATCH --mem=32G
#SBATCH --gpus=1

# run my job (some executable)
python3 full_inference.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace