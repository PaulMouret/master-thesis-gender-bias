from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys

from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile, file_already_exists
from padding_analysis.padding_analysis_utils import get_llama_padded_strings_pronoun_probablities, res_analysis, \
    abs_log_bias, sum_relevance
from utils.constants_utils import *


context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[0]

my_dataset = load_dataset('json', data_files=f'../created_datasets/{context_dataset_name}.jsonl',
                          split='train')
subset = my_dataset.train_test_split(test_size=min(subset_size, my_dataset.shape[0] - 1),
                                     seed=random_seed)['test']

strings = subset['text']
padding_policies = ["bos"]
# "pad" turns out to be of no use at the end, because the model was not trained with it so it gives bad results
# "eos" doesn't give as bad results as pad, but actually thanks to attention masks the padding character does
# not really matter, so, in order to save computation time, we discard it

# 1. We create the results of padded contexts
for padding_policy in padding_policies:
    if not file_already_exists("../saved_objects/padding_analysis/padded_contexts_inference/",
                               f"{context_dataset_name}_padding_{padding_policy}_res.json"):
        res = get_llama_padded_strings_pronoun_probablities(strings, 10, pad_token=padding_policy, verbose=True)
        my_dict = {'res': res}
        store_obj_in_jsonfile(my_dict, "../saved_objects/padding_analysis/padded_contexts_inference/",
                              f"{context_dataset_name}_padding_{padding_policy}_res")

# 2. We analyze the results (and create corresponding objects)
for padding_policy in padding_policies:
    res = load_obj_from_jsonfile("../saved_objects/padding_analysis/padded_contexts_inference/",
                                 f"{context_dataset_name}_padding_{padding_policy}_res")['res']
    res_analysis(res, abs_log_bias, sum_relevance, context_dataset_name, addon_name=padding_policy)
    #Note that we use abs_log_bias instead of log_bias, because in order for our relative_bias_difference
    #to be meaningful, it should correspond to a positive measure
