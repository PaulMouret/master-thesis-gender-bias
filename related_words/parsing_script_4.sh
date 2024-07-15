#!/bin/bash

#SBATCH -J parsing_script_4					  # name of job
#SBATCH -c1				  # name of partition or queue
#SBATCH -o parsing_script_4.out				  # name of output file for this submission script
#SBATCH -e parsing_script_4.err				  # name of error file for this submission script

curl -F data=@../created_datasets/related_words/contexts_to_parse_200000_4.txt -F model=english -F input=horizontal -F tagger= -F parser= http://lindat.mff.cuni.cz/services/udpipe/api/process -o ../created_datasets/related_words/parsed_contexts_200000_4.json
