from datasets import load_dataset
from blingfire import text_to_words
import math

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

# 2. We tokenize each (original) sentence of the dataset, and store it

#Because the ud pipe server can not receive too many requests at once
data_size = len(my_dataset)
parsed_size = 20000

for i in range(math.ceil(data_size / parsed_size)):
    a = i * parsed_size
    b = min((i + 1) * parsed_size, data_size)
    subset = my_dataset["original_text"][a:b]

    with open(f'../created_datasets/related_words/contexts_to_parse_{subset_size}_{i}.txt', 'w', encoding="utf-8") as f:
        for example in subset:
            tokenized_sentence = ' '.join(text_to_words(example).split())
            f.write(tokenized_sentence + "\n")
        f.close()

#The command line to process it, from the directory of contexts_to_parse, is :
#curl -F data=@contexts_to_parse.txt -F model=english -F input=horizontal -F tagger= -F parser= http://lindat.mff.cuni.cz/services/udpipe/api/process -o parsed_contexts.json
