from datasets import load_dataset
from blingfire import text_to_words

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.inference_utils import load_tokenizer
from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile
from utils.noising_utils import find_token_range_from_strings
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


tokenizer = load_tokenizer(model_name)

stereotyped_ls = []
stereotyped_related_ls = []
for example in full_dataset.to_iterable_dataset():
    sentence = example['text']
    tokenized_sentence = text_to_words(sentence)
    index_word = 0
    for i_word, word in enumerate(tokenized_sentence.split()):
        related = example["related_words"][i_word]
        stereotyped, male_stereotyped = word_is_stereotyped(word)
        if stereotyped:
            tok_start_end = find_token_range_from_strings(tokenizer, sentence, word, index_word)
            knowledge = {'prompt': sentence, 'subject': word, 'subject_index': index_word,
                         'subject_start_end': tok_start_end, "subject_male_stereotyped": male_stereotyped,
                         'subject_related': related,
                         #in case we need them for analysis, we also include other characteristics
                         "dialogue_context": example["dialogue_context"],
                         "prob_he": example["prob_he"], "prob_she": example["prob_she"],
                         "relevance_score": example["relevance_score"], "bias_score": example["bias_score"],
                         "male_stereotype": example["male_stereotype"]}
            stereotyped_ls.append(knowledge)
            if related:
                stereotyped_related_ls.append(knowledge)
        index_word += len(word)  #we don't add 1 for the space because of how find_token_range is implemented

store_obj_in_jsonfile(stereotyped_ls, '../created_datasets/causal_tracing_datasets/', f"stereotyped_{subset_size}")
store_obj_in_jsonfile(stereotyped_related_ls, '../created_datasets/causal_tracing_datasets/',
                      f"stereotyped_related_{subset_size}")
