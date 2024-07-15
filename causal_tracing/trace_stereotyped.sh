#!/bin/bash

#SBATCH -J trace_stereotyped					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o trace_stereotyped.out				  # name of output file for this submission script
#SBATCH -e trace_stereotyped.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -G nvidia_geforce_rtx_3090:1

# run my job (some executable)
python3 trace.py --model_name_path "meta-llama/Llama-2-7b-hf" --known_dataset "stereotyped_200000" --noise_level 0.0424 --cap_examples 1000

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace