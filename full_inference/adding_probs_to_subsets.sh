#!/bin/bash

#SBATCH -J adding_probs_to_subsets					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o adding_probs_to_subsets.out				  # name of output file for this submission script
#SBATCH -e adding_probs_to_subsets.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 adding_probs_to_subsets.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace