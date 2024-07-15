#!/bin/bash

#SBATCH -J format_parsed_contexts					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o format_parsed_contexts.out				  # name of output file for this submission script
#SBATCH -e format_parsed_contexts.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 format_parsed_contexts.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
