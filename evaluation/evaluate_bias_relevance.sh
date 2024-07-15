#!/bin/bash

#SBATCH -J evaluate_bias_relevance					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o evaluate_bias_relevance.out				  # name of output file for this submission script
#SBATCH -e evaluate_bias_relevance.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 evaluate_bias_relevance.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace