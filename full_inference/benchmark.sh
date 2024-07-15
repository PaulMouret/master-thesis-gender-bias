#!/bin/bash

#SBATCH -J benchmark					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o benchmark.out				  # name of output file for this submission script
#SBATCH -e benchmark.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 benchmark.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace