#!/bin/bash

#SBATCH -J test_hypothesis_emb_variance					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o test_hypothesis_emb_variance.out				  # name of output file for this submission script
#SBATCH -e test_hypothesis_emb_variance.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=32G
#SBATCH --gpus=1

# run my job (some executable)
python3 test_hypothesis_emb_variance.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace