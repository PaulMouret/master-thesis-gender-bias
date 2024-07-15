#!/bin/bash

#SBATCH -J all_full_inference					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o all_full_inference.out				  # name of output file for this submission script
#SBATCH -e all_full_inference.err				  # name of error file for this submission script
#SBATCH -G nvidia_geforce_rtx_3090:1

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint="gpuram24G"

# run my job (some executable)
python3 adding_probs_to_subsets.py
python3 benchmark.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace