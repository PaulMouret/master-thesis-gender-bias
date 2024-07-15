from datasets import load_dataset
import numpy as np

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from benchmark_utils import full_inference_analysis
from utils.constants_utils import *


model_names = MODEL_NAMES
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = 50000#SUBSET_SIZES[-1]

for model_name in model_names:
    jsonl_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}" \
                 f"_{'_'.join(model_name.split('/')[-1].split())}"
    #The .join is necessary for finetuned models where space in the name ;
    # but it shouldn't change anything compared to old code for base model

    #We don't need the original dataset inference was obtained from, except for its length :
    my_dataset = load_dataset('json', data_files=f'../created_datasets/inferred_datasets/'
                                                 f'{jsonl_name}.jsonl',
                              split='train')

    #We print the contexts with minimum and maximum biases
    all_biases = my_dataset["bias_score"]
    i_max = np.argmax(all_biases)
    i_min = np.argmin(all_biases)
    print(f"{my_dataset['text'][i_max]} ({i_max}) : bias {all_biases[i_max]}")
    print(f"{my_dataset['text'][i_min]} ({i_min}) : bias {all_biases[i_min]}")

    for subdataset_name in [jsonl_name, "dialogue_"+jsonl_name, "classical_"+jsonl_name]:
        if subdataset_name == "dialogue_"+jsonl_name:
            subdataset = my_dataset.filter(lambda x: x["dialogue_context"])
        elif subdataset_name == "classical_"+jsonl_name:
            subdataset = my_dataset.filter(lambda x: not x["dialogue_context"])
        elif subdataset_name == jsonl_name:
            subdataset = my_dataset
        else:
            raise Exception(f"Unknown subdataset name : {subdataset_name}")

        lengths = [len(example['text'].split()) for example in subdataset]

        print(f"\n################\n{subdataset_name}\n")
        full_inference_analysis(subdataset['prob_he'], subdataset['prob_she'], subdataset['bias_score'],
                                subdataset['relevance_score'], lengths, subdataset_name)
