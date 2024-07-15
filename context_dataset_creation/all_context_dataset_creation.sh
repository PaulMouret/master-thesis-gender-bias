#!/bin/bash

#SBATCH -J all_context_dataset_creation					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o all_context_dataset_creation.out				  # name of output file for this submission script
#SBATCH -e all_context_dataset_creation.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 gender_of_persons.py
python3 firstnames.py
python3 manual_gendered_words.py
python3 stereotyped_occupations.py

python3 creating_dataset.py

python3 statistics_visualization.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
