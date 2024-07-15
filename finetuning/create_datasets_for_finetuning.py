from datasets import load_dataset, Dataset
import numpy as np
import pyarrow
from blingfire import text_to_words

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile
from utils.constants_utils import *


# 0. We load the base dataset

model_name = BASE_MODEL_NAME

context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]

full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

full_dataset = load_dataset('json',
                            data_files=f'../created_datasets/inferred_datasets/{full_dataset_name}.jsonl',
                            split='train')

# 1. Relevance
# We will build several datasets, but they all consist in relevant contexts
relevance_threshold = RELEVANCE_THRESHOLD
full_dataset = full_dataset.filter(lambda x: x["relevance_score"] >= relevance_threshold)

# 2. Common test set (of contexts, ie. without pronoun at the end)
train_test = full_dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = train_test['test']
test_dataset.to_json(f'../created_datasets/finetuning_datasets/test_{full_dataset_name}.jsonl')

# 3. We create the different training sets (that include a possible validation set)

train_dataset = train_test['train']

# a. To manage the stereotyped datasets

# We load the dictionaries
dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                        f"dict_bias_diff_{subset_size}")
dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                     f"dict_counts_{subset_size}")
dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                   f"dict_stereotypical_scores_{subset_size}")

frequency_threshold = OCCURRENCE_THRESHOLD
male_threshold = MALE_THRESHOLD
female_threshold = FEMALE_THRESHOLD

def stereotype_in_context(context):
    #The function is built so that we can use the dictionaries on datasets they were not computed from,
    #ie. where some words of the sentences are not in the dictionaries
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()
              if word in dict_stereotypical_scores.keys() and dict_counts[word] >= frequency_threshold]
    if scores: #if the list is not empty
        female_score, male_score = min(scores), max(scores)
        return (female_score < female_threshold) or (male_score > male_threshold)
    else:
        return False

# b. We go through all possible kinds of datasets
choices_stereotypical = [True, False]
strategies = FINETUNING_STRATEGIES #strategies for choosing the target pronoun

finetuning_random_seed = FINETUNING_RANDOM_SEED
LLAMA_PRONOUNS = {"pos": "he",
                  "neg": "she",
                  "neut": "they"}

# We want comparable results, so all finetuning datasets must have same size
# But, there are less stereotyped contexts than contexts, so we will truncate all finetuning datasets with this number
min_size = len(train_dataset.filter(lambda x: stereotype_in_context(x)))
print(f"There are {min_size} stereotyped contexts in training set.")

for choice_stereotypical in choices_stereotypical:
    if choice_stereotypical:
        finetuning_dataset = train_dataset.filter(lambda x: stereotype_in_context(x))
        stereotype_str = "stereotyped"
    else:
        finetuning_dataset = train_dataset
        stereotype_str = "nonstereotyped"

    #We truncate it with the correct size
    finetuning_dataset = finetuning_dataset.filter(lambda x, i: i < min_size, with_indices=True)

    for strategy in strategies:
        np.random.seed(finetuning_random_seed)
        list_of_texts = []
        for example in finetuning_dataset:
            if strategy == "random":
                target = np.random.choice(list(LLAMA_PRONOUNS.values()))
            elif strategy == "neutral":
                target = LLAMA_PRONOUNS["neut"]
            elif strategy == "opposite":
                if example["male_stereotype"]:
                    target = LLAMA_PRONOUNS["neg"]
                else:
                    target = LLAMA_PRONOUNS["pos"]
            else:
                raise Exception(f"Unknown strategy {strategy}")

            list_of_texts.append(example["text"] + " " + target)

        stored_dataset = Dataset(pyarrow.Table.from_arrays([list_of_texts], names=["text"]))
        stored_dataset.to_json(f'../created_datasets/finetuning_datasets/'
                               f'train_{stereotype_str}_{strategy}_{full_dataset_name}.jsonl')
