from datasets import load_dataset
import json
import math

import sys

from utils.constants_utils import *


# 1. We load the context dataset
context_dataset_name = CONTEXT_DATASET_NAME
model_name = BASE_MODEL_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
my_dataset = load_dataset('json', data_files=f'../created_datasets/noising_experiment_datasets/'
                                             f'{full_dataset_name}_final_noising_experiment.jsonl',
                          split='train')

data_size = len(my_dataset)
parsed_size = 20000

list_conllu_str = []
for i in range(math.ceil(data_size / parsed_size)):
    # 2. We load the parsed contexts
    json_filename = f"../created_datasets/related_words/parsed_contexts_{subset_size}_{i}.json"
    #One should be careful to the encoding
    with open(json_filename, 'r', encoding="utf-8") as f:
        obj = json.load(f)
        result = obj["result"].rstrip("\n")
        f.close()

    # 3. We add the parsing to the dataset
    # It may not be optimal in terms of storage, but its much more practical
    list_conllu_str += result.split("\n\n")

my_dataset = my_dataset.add_column("parsed_original_text", list_conllu_str)
#We save it
my_dataset.to_json(f'../created_datasets/noising_experiment_datasets/'
                   f'{full_dataset_name}_final_noising_experiment.jsonl')
