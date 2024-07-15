#!/bin/bash

#SBATCH -J verbose_related_words					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o verbose_related_words.out				  # name of output file for this submission script
#SBATCH -e verbose_related_words.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 verbose_related_words.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
