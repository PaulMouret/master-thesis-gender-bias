from datasets import load_dataset
import time
import numpy as np
from blingfire import text_to_words

import sys

from utils.constants_utils import *


# 1. We load the dataset
context_dataset_name = CONTEXT_DATASET_NAME
model_name = BASE_MODEL_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
my_dataset = load_dataset('json', data_files=f'../created_datasets/noising_experiment_datasets/'
                                             f'{full_dataset_name}_final_noising_experiment.jsonl',
                          split='train')

# 2. We apply the mapping function (and create a new "TEST" dataset file)
# There is no significative difference between cpu and gpu : it takes between 2 and 3s per example
for example in my_dataset:
    if np.count_nonzero(example['related_words']) >= 3:
        print(f"\n{example['text']}\n"
              f"related words : "
              f"{np.array(text_to_words(example['text']).split())[np.array(example['related_words'])]}")

#my_dataset = my_dataset.map(related_words_mapping_function, fn_kwargs={"verbose": True})
#print(f"\nMapping done")
#my_dataset.to_json(f'../created_datasets/noising_experiment_datasets/'
                   #f'TEST_{full_dataset_name}_final_noising_experiment.jsonl')
