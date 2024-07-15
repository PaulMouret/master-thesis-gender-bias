#!/bin/bash

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=paul.mouret2000@gmail.com

#SBATCH -J creating_other_datasets					  # name of job
#SBATCH -p gpu					  # name of partition or queue
#SBATCH -o creating_other_datasets.out				  # name of output file for this submission script
#SBATCH -e creating_other_datasets.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G
#SBATCH --gpus=1

# run my job (some executable)
python3 creating_other_datasets.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
