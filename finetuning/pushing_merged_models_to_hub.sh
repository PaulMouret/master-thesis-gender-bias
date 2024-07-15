#!/bin/bash

#SBATCH -J pushing_merged_models_to_hub					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o pushing_merged_models_to_hub.out				  # name of output file for this submission script
#SBATCH -e pushing_merged_models_to_hub.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 pushing_merged_models_to_hub.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace