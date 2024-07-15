from datasets import load_dataset
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr
from collections import Counter
from blingfire import text_to_words

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import *
from utils.constants_utils import *


# I - Creation of useful objects

# 0. Load dataset

model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

corrupted_dataset_name = f'{full_dataset_name}_final_noising_experiment'

corrupted_dataset = load_dataset('json',
                                 data_files=f'../created_datasets/noising_experiment_datasets/'
                                            f'{corrupted_dataset_name}.jsonl',
                                 split='train')

# 1. We load the dictionaries
dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                        f"dict_bias_diff_{subset_size}")
dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                     f"dict_counts_{subset_size}")
dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                   f"dict_stereotypical_scores_{subset_size}")

# II - Looking for sentences

# 1. paradoxical
def all_feminine_stereotyped(context):
    counts = [dict_counts[word] for word in text_to_words(context["text"]).lower().split()]
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()]
    return min(counts) >= OCCURRENCE_THRESHOLD and max(scores) <= 0


paradox_dataset = corrupted_dataset.filter(lambda context: all_feminine_stereotyped(context)
                                                           and context['bias_score'] > 0)

#2. kid related
def min_counts_in_context(context):
    return min([dict_counts[word] for word in text_to_words(context["text"]).lower().split()])

frequent_dataset = corrupted_dataset.filter(lambda x: min_counts_in_context(x) >= OCCURRENCE_THRESHOLD)

def opposite_stereotypes_in_context(context):
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()]
    return min(scores) <= FEMALE_THRESHOLD and max(scores) >= MALE_THRESHOLD

opposite_dataset = frequent_dataset.filter(lambda x: opposite_stereotypes_in_context(x))

for c in opposite_dataset['text']:
    print(c)

def neutral_words_in_context(context):
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()]
    return min(scores) >= FEMALE_THRESHOLD/2 and max(scores) <= MALE_THRESHOLD/2

neutral_dataset = frequent_dataset.filter(lambda x: neutral_words_in_context(x))

#for c in neutral_dataset['text']:
    #print(c)

def related_kid_in_context(context):
    words = [word for word in text_to_words(context["text"]).lower().split()]
    return "kid" in words and context['related_words'][words.index("kid")]

related_kid_dataset = frequent_dataset.filter(lambda x: related_kid_in_context(x))
for c in related_kid_dataset['text']:
    print(c)
