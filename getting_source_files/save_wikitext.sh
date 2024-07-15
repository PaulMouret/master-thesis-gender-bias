#!/bin/bash

#SBATCH -J save_wikitext					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o save_wikitext.out				  # name of output file for this submission script
#SBATCH -e save_wikitext.err				  # name of error file for this submission script

#specific "capacity" commands
#SBATCH --mem=64G

# run my job (some executable)
python3 save_wikitext.py

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
