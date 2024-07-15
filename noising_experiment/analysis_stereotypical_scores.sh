#!/bin/bash

#SBATCH -J analysis_stereotypical_scores				  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o analysis_stereotypical_scores.out				  # name of output file for this submission script
#SBATCH -e analysis_stereotypical_scores.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 analysis_stereotypical_scores.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
