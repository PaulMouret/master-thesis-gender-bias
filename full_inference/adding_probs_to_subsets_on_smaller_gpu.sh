#!/bin/bash

#SBATCH -J adding_probs_to_subsets_on_smaller_gpu					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o adding_probs_to_subsets_on_smaller_gpu.out				  # name of output file for this submission script
#SBATCH -e adding_probs_to_subsets_on_smaller_gpu.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 adding_probs_to_subsets_on_smaller_gpu.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace