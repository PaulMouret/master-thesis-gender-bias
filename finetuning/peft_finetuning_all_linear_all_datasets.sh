#!/bin/bash

#SBATCH -J peft_finetuning_all_linear_all_datasets					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o peft_finetuning_all_linear_all_datasets.out				  # name of output file for this submission script
#SBATCH -e peft_finetuning_all_linear_all_datasets.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 peft_finetuning_all_linear_all_datasets.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace