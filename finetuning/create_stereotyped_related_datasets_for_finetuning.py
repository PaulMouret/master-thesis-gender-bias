from datasets import load_dataset, Dataset
from blingfire import text_to_words
import numpy as np
import pyarrow

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile
from utils.constants_utils import *


context_dataset_name = CONTEXT_DATASET_NAME
model_name = BASE_MODEL_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
full_dataset = load_dataset('json', data_files=f'../created_datasets/noising_experiment_datasets/'
                                               f'{full_dataset_name}_final_noising_experiment.jsonl',
                            split='train')

# 1. Relevance
relevance_threshold = RELEVANCE_THRESHOLD
#full_dataset = full_dataset.filter(lambda x: x["relevance_score"] is not None)
# to avoid problems, but shouldn't be necessary
full_dataset = full_dataset.filter(lambda x: x["relevance_score"] >= relevance_threshold)
#full_dataset = full_dataset.filter(lambda x, i: i < max_contexts, with_indices=True)

# 2. Stereotypes
dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                        f"dict_bias_diff_{subset_size}")
dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                     f"dict_counts_{subset_size}")
dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                   f"dict_stereotypical_scores_{subset_size}")

frequency_threshold = OCCURRENCE_THRESHOLD
male_threshold = MALE_THRESHOLD
female_threshold = FEMALE_THRESHOLD


def word_is_stereotyped(w):
    lw = w.lower()
    if lw in dict_stereotypical_scores.keys():
        if dict_counts[lw] >= frequency_threshold:
            score = dict_stereotypical_scores[lw]
            return (score < female_threshold) or (score > male_threshold), (score > male_threshold)
    return False, False


def stereotyped_mapping_function(example):
    sentence = example['text']
    tokenized_sentence = text_to_words(sentence)
    stereotyped_example = False
    for i_word, word in enumerate(tokenized_sentence.split()):
        stereotyped, male_stereotyped = word_is_stereotyped(word)
        stereotyped_example = stereotyped_example or stereotyped
    return stereotyped_example


def stereotyped_related_mapping_function(example):
    sentence = example['text']
    tokenized_sentence = text_to_words(sentence)
    stereotype_related_example = False
    for i_word, word in enumerate(tokenized_sentence.split()):
        related = example["related_words"][i_word]
        stereotyped, male_stereotyped = word_is_stereotyped(word)
        stereotype_related_example = stereotype_related_example or (related and stereotyped)
    return stereotype_related_example


train_test = full_dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = train_test['test']

#Now we have the test set :
# to make it lighter and prevent from subsequent problems, we remove already computed attributes
potential_columns_to_remove = ["corrupted_probs_he", "corrupted_probs_she",
                                   "corrupted_relevances", "corrupted_biases",
                                   "bias_differences", "relevance_differences", "related_words",
                                   "prob_he", "prob_she", "relevance_score", "bias_score", "male_stereotype",
                                   "parsed_original_text"]
for colname in potential_columns_to_remove:
    if colname in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(colname)
#Now we can save it
test_dataset.to_json(f'../created_datasets/finetuning_datasets/test_ft.jsonl')

train_dataset = train_test['train']
min_size = len(train_dataset.filter(lambda x: stereotyped_related_mapping_function(x)))
# to have all datasets of the same size

stereotyped_finetuning_dataset = train_dataset.filter(lambda x: stereotyped_mapping_function(x))
stereotyped_finetuning_dataset = stereotyped_finetuning_dataset.filter(lambda x, i: i < min_size, with_indices=True)

stereotyped_related_finetuning_dataset = train_dataset.filter(lambda x: stereotyped_related_mapping_function(x))
stereotyped_related_finetuning_dataset = stereotyped_related_finetuning_dataset.filter(lambda x, i: i < min_size,
                                                                                       with_indices=True)

#From there, by the random strategy, we can create the datasets for finetuning and dama
# (they are intrisically the same, but their form is not the same)
LLAMA_PRONOUNS = {"pos": "he",
                  "neg": "she",
                  "neut": "they"}

for dataset, dataset_desc in zip([stereotyped_finetuning_dataset, stereotyped_related_finetuning_dataset],
                                 ["stereotyped", "stereotyped_related"]):
    np.random.seed(42)
    target_list = [np.random.choice(list(LLAMA_PRONOUNS.values())) for i in range(len(dataset))]

    list_of_texts = []
    list_of_dicts = []
    for example, target in zip(dataset, target_list):
        # a. The transformers dataset
        list_of_texts.append(example["text"] + " " + target)
        # b. The json dataset for DAMA
        list_of_dicts.append({"prompt": example["text"], "subject": "", "target_new": {"str": target}})

    stored_dataset = Dataset(pyarrow.Table.from_arrays([list_of_texts], names=["text"]))
    stored_dataset.to_json(f'../created_datasets/finetuning_datasets/'
                           f'train_ft_{dataset_desc}.jsonl')

    store_obj_in_jsonfile(list_of_dicts, '../created_datasets/finetuning_datasets/', f'train_dama_{dataset_desc}')
