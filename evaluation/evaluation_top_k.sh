#!/bin/bash

#SBATCH -J evaluation_top_k					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o evaluation_top_k.out				  # name of output file for this submission script
#SBATCH -e evaluation_top_k.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 evaluation_top_k.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace