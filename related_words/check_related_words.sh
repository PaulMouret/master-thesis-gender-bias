#!/bin/bash

#SBATCH -J check_related_words					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o check_related_words.out				  # name of output file for this submission script
#SBATCH -e check_related_words.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 check_related_words.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
