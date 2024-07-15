from datasets import load_dataset
import numpy as np

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.constants_utils import *


# I - Creation of useful objects

# 0. Load dataset

causal_trace_dataset = load_dataset('json',
                                    data_files=f'../created_datasets/causal_tracing_datasets/'
                                               f'results_stereotyped_200000_1000.jsonl',
                                    split='train')
n = len(causal_trace_dataset)
#subject_male_stereotyped_dataset = causal_trace_dataset.filter(lambda x: x['subject_male_stereotyped'])
#subject_related_dataset = causal_trace_dataset.filter(lambda x: x['subject_related'])
#dialogue_context_dataset = causal_trace_dataset.filter(lambda x: x['dialogue_context'])

#print(f"\nproportion suject male : {round(100 * len(subject_male_stereotyped_dataset) / n, 2)} %"
#      f"\nproportion subject related : {round(100 * len(subject_related_dataset) / n, 2)} %"
#      f"\nproportion dialogue context : {round(100 * len(dialogue_context_dataset) / n, 2)} %")

subject_start_ends = causal_trace_dataset["subject_start_end"]
lengths = [l[1] - l[0] for l in subject_start_ends]

for length in np.arange(1, 5):
    nb_occurrences = lengths.count(length)
    print(f"length {length} : {round(100 * nb_occurrences / n, 2)}")
