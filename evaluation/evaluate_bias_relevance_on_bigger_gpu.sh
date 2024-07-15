#!/bin/bash

#SBATCH -J evaluate_bias_relevance_on_bigger_gpu					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o evaluate_bias_relevance_on_bigger_gpu.out				  # name of output file for this submission script
#SBATCH -e evaluate_bias_relevance_on_bigger_gpu.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 evaluate_bias_relevance.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace