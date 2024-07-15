import numpy as np
from datasets import load_dataset
from blingfire import text_to_words

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile
from utils.constants_utils import *


#To remove contexts that contain a word specified in manual_gendered
#This program aims to avoid reconstructing the whole dataset, when we just remove contexts, and just a few of them
#However, note that, this way, statistics about the number of contexts are not the same anymore ;
#for this the whole dataset creation should be rerun

model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
corrupted_dataset_name = f'{full_dataset_name}_final_noising_experiment'
jsonl_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}" \
                     f"_{'_'.join(model_name.split('/')[-1].split())}"

datasets_to_update = [#f'../created_datasets/{context_dataset_name}',
                      f'../created_datasets/noising_experiment_datasets/{corrupted_dataset_name}',
                      f"../created_datasets/inferred_datasets/{jsonl_name}"]

for dataset_to_update in datasets_to_update:
    print(f"\n{dataset_to_update}")
    my_dataset = load_dataset('json', data_files=f'{dataset_to_update}.jsonl',
                              split='train')
    print(f"The original dataset has length {len(my_dataset)}")

    manual_gendered_persons = load_obj_from_jsonfile("../created_datasets/utils_datasets/", "manual_gendered_persons")
    manual_gendered_persons = set(manual_gendered_persons)
    my_dataset = my_dataset.filter(
        lambda x: not any({word in manual_gendered_persons
                           for word in text_to_words(x['text'].lower()).split()}))

    manual_gendered_firstnames = load_obj_from_jsonfile("../created_datasets/utils_datasets/",
                                                        "manual_gendered_firstnames")
    manual_gendered_firstnames = set(manual_gendered_firstnames)
    my_dataset = my_dataset.filter(
        lambda x: not any({word in manual_gendered_firstnames for word in text_to_words(x['text']).split()}))
    print("Filtered factually gendered first names")

    print(f"The new dataset has length {len(my_dataset)}")

    my_dataset.to_json(f'{dataset_to_update}.jsonl')
