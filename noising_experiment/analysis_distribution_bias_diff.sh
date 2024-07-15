#!/bin/bash

#SBATCH -J analysis_distribution_bias_diff				  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o analysis_distribution_bias_diff.out				  # name of output file for this submission script
#SBATCH -e analysis_distribution_bias_diff.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 analysis_distribution_bias_diff.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace