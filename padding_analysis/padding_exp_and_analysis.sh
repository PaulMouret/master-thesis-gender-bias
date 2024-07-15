#!/bin/bash

#SBATCH -J padding_exp_and_analysis					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o padding_exp_and_analysis.out				  # name of output file for this submission script
#SBATCH -e padding_exp_and_analysis.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 padding_exp_and_analysis.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace