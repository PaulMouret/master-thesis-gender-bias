#!/bin/bash

#SBATCH -J statistics_visualization					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o statistics_visualization.out				  # name of output file for this submission script
#SBATCH -e statistics_visualization.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 statistics_visualization.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
