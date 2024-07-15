#!/bin/bash

#SBATCH -J final_noising_experiment_on_bigger_gpu					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o final_noising_experiment_on_bigger_gpu.out				  # name of output file for this submission script
#SBATCH -e final_noising_experiment_on_bigger_gpu.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 final_noising_experiment.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace