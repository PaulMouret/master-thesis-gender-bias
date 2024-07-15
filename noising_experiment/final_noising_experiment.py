from datasets import load_dataset
import torch
import time
from blingfire import text_to_words

import sys

from utils.global_utils import relevance_score, bias_score
from utils.noising_utils import calculate_corrupted_scores, ModelAndTokenizer
from utils.constants_utils import *


def dataset_mapping_function(example):
    #Once we have the probabilities for 'he' and 'she', it is easy to compute all these values ;
    #the simplest seems to store it directly in the dataset instead of computing it everytime
    corrupted_relevances = []
    corrupted_biases = []
    bias_differences = []
    relevance_differences = []
    #no need of biases and relevance differences at this point, as we are only interested in relevances
    for i in range(len(example["corrupted_probs_he"])):
        corrupted_relevance = relevance_score(example["corrupted_probs_he"][i], example["corrupted_probs_she"][i])
        corrupted_bias = bias_score(example["corrupted_probs_he"][i], example["corrupted_probs_she"][i])
        corrupted_relevances.append(corrupted_relevance)
        corrupted_biases.append(corrupted_bias)
        bias_differences.append(example["bias_score"] - corrupted_bias)
        relevance_differences.append(example["relevance_score"] - corrupted_relevance)
    return {"corrupted_relevances": corrupted_relevances, "corrupted_biases": corrupted_biases,
            "bias_differences": bias_differences, "relevance_differences": relevance_differences}


print("Loading model")
model_start = time.time()
model_name = BASE_MODEL_NAME
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
mt = ModelAndTokenizer(model_name, torch_dtype=torch_dtype)
model_end = time.time()
print("Model loaded")
print(f"model loading time : {model_end - model_start}")

context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]

full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

full_dataset = load_dataset('json',
                            data_files=f'../created_datasets/inferred_datasets/{full_dataset_name}.jsonl',
                            split='train')

relevance_threshold = RELEVANCE_THRESHOLD
#full_dataset = full_dataset.filter(lambda x: x["relevance_score"] is not None)
# to avoid problems, but shouldn't be necessary
full_dataset = full_dataset.filter(lambda x: x["relevance_score"] >= relevance_threshold)
#full_dataset = full_dataset.filter(lambda x, i: i < max_contexts, with_indices=True)
final_nb_contexts = len(full_dataset)
print(f"Dataset loaded : {final_nb_contexts} at the end")
#Count 7s for 1 context, ie. 1 hour for 500 contexts

factor = FINAL_FACTOR
final_std = FINAL_STD

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
        index_word += len(word)  #we don't add 1 for the space because of how find_token_range is implemented
    corrupted_probs_he.append(c_probs_he)
    corrupted_probs_she.append(c_probs_she)

factor_dataset = full_dataset.add_column("corrupted_probs_he", corrupted_probs_he)
factor_dataset = factor_dataset.add_column("corrupted_probs_she", corrupted_probs_she)
#A mapping function for biases, relevances and their differences
#(because corrupted probs
factor_dataset = factor_dataset.map(dataset_mapping_function)
factor_dataset.to_json(f'../created_datasets/noising_experiment_datasets/'
                       f'{full_dataset_name}_final_noising_experiment.jsonl')
end = time.time()

print(f"Time needed : {end - start}")
print(f"Average time needed for one context : {(end - start)/final_nb_contexts}")
