#!/bin/bash

#SBATCH -J all_final_noising_experiment_analyses				  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o all_final_noising_experiment_analyses.out				  # name of output file for this submission script
#SBATCH -e all_final_noising_experiment_analyses.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 analysis_final_noising_experiment.py
python3 analysis_distribution_bias_diff.py
python3 analysis_stereotypical_scores.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace