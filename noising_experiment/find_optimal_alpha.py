from datasets import load_dataset
import torch
import time
from blingfire import text_to_words
import numpy as np

import sys

from utils.global_utils import relevance_score, bias_score
from utils.noising_utils import calculate_corrupted_scores, ModelAndTokenizer
from utils.constants_utils import *


def factor_dataset_mapping_function(example):
    corrupted_relevances = []
    #no need of biases and relevance differences at this point, as we are only interested in relevances
    for i in range(len(example["corrupted_probs_he"])):
        corrupted_relevances.append(relevance_score(example["corrupted_probs_he"][i], example["corrupted_probs_she"][i]))
    return {"corrupted_relevances": corrupted_relevances}


model_start = time.time()
model_name = BASE_MODEL_NAME
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
mt = ModelAndTokenizer(model_name, torch_dtype=torch_dtype)
model_end = time.time()
print("Model loaded")
print(f"model loading time : {model_end - model_start}")

context_dataset_name = CONTEXT_DATASET_NAME
subset_size = 50000
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE

full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
full_dataset = load_dataset('json',
                            data_files=f'../created_datasets/inferred_datasets/{full_dataset_name}.jsonl',
                            split='train')

relevance_threshold = RELEVANCE_THRESHOLD
full_dataset = full_dataset.filter(lambda x: x["relevance_score"] >= relevance_threshold)
max_contexts = 1000 #because of computation time, we have to set a limit
full_dataset = full_dataset.filter(lambda x, i: i < max_contexts, with_indices=True)
print("Dataset loaded")

factors = list(np.arange(1, 5.1, 0.1)) + [100.]
factors = [round(x, 2) for x in factors]
#it takes about 63000 s to test one factor, ie. 17.5 hours (for 150 contexts = 1600 noised contets)
#factors = [3.2, 3.4, 3.6, 3.8] #is another experiment I thought about, but it may not be necessary
final_std = FINAL_STD
#We create  dataset for each factor

global_start = time.time()
for factor in factors:
    start = time.time()
    corrupted_probs_he = []
    corrupted_probs_she = []
    for example in full_dataset.to_iterable_dataset():
        sentence = example['text']
        tokenized_sentence = text_to_words(sentence)
        c_probs_he = []
        c_probs_she = []
        index_word = 0
        for i_word, word in enumerate(tokenized_sentence.split()):
            c_probabilities = calculate_corrupted_scores(
                mt,
                sentence,
                word,
                index_word,
                samples=10,
                noise=factor * final_std,
            )
            c_probabilities = c_probabilities.cpu().numpy().tolist()
            c_probs_he.append(c_probabilities[0])
            c_probs_she.append(c_probabilities[1])
            index_word += len(word)  # we don't add 1 for the space because of how find_token_range is implemented
        corrupted_probs_he.append(c_probs_he)
        corrupted_probs_she.append(c_probs_she)

    factor_dataset = full_dataset.add_column("corrupted_probs_he", corrupted_probs_he)
    factor_dataset = factor_dataset.add_column("corrupted_probs_she", corrupted_probs_she)
    #A mapping function for biases, relevances and their differences
    #(because corrupted probs
    factor_dataset = factor_dataset.map(factor_dataset_mapping_function)
    factor_dataset.to_json(f'../created_datasets/noising_experiment_datasets/'
                           f'{full_dataset_name}_{max_contexts}_corrupted_factor_{factor}.jsonl')
    end = time.time()

    print("factor done")
    print(f"Time needed for factor {factor} : {end - start}")

global_end = time.time()
print(f"\ntotal time needed for {len(factors)} factors and {max_contexts} contexts : {global_end - global_start} s.")
