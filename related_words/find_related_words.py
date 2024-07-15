from datasets import load_dataset

import sys

from utils.constants_utils import *
from find_related_words_utils import related_words_mapping_function


# 1. We load the dataset
context_dataset_name = CONTEXT_DATASET_NAME
model_name = BASE_MODEL_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
my_dataset = load_dataset('json', data_files=f'../created_datasets/noising_experiment_datasets/'
                                             f'{full_dataset_name}_final_noising_experiment.jsonl',
                          split='train')
# For debugging :
#max_contexts = 100
#my_dataset = my_dataset.filter(lambda x, i: i < max_contexts, with_indices=True)

# 2. We apply the mapping function (and replace the original dataset file)
# There is no significative difference between cpu and gpu : it takes between 2 and 3s per example
my_dataset = my_dataset.map(related_words_mapping_function)
my_dataset.to_json(f'../created_datasets/noising_experiment_datasets/'
                   f'{full_dataset_name}_final_noising_experiment.jsonl')
