from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr
from collections import Counter
from blingfire import text_to_words
from prettytable import PrettyTable

import sys

from utils.global_utils import load_obj_from_jsonfile, print_full_descriptive_statistics, prettytable_to_latex
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

# 2. Utils
def min_counts_in_context(context):
    return min([dict_counts[word] for word in text_to_words(context["text"]).lower().split()])

corrupted_dataset = corrupted_dataset.filter(lambda x: min_counts_in_context(x) >= OCCURRENCE_THRESHOLD)


examples1 = ["\"Don't sell yourself short, kid,\"",
            "Maybe it's some kid who thought",
            "Grabbing the kid's arm,",
            "One kid had been so nervous",
            "\"What have you kids been feeding back there?\"",
            "\"Only one of those kids is mine!\"",
            "With the kids on the bus,",]

examples2 = ["18 Also day by day, from the first day unto the last day,",
            "\"Don't work too hard on your lipstick,\"",
            "Behind those red covered lips in lipstick,",
             "Downstairs, the security guard hurried from behind the booth when",
             "The guard's eyes widened as",
             "\"A nightmare, eh?\"",
             "\"Why are we leaving the party so early?\"",
             "With blue eyes as pale as the rain,"]


for id_list, examples_list in enumerate((examples1, examples2)):

    examples_dataset = corrupted_dataset.filter(lambda x: x["text"] in examples_list)

    results_directory = "../saved_objects/noising_experiment/related_words/"
    tabout = open(f"{results_directory}results_latex_{id_list}.txt", 'w')

    for example in examples_list:
        single_dataset = examples_dataset.filter(lambda x: x["text"] == example)
        if len(single_dataset) > 0:
            context = single_dataset[0]
            context_list = text_to_words(context['text']).split()
            related_words = context['related_words']
            stereotypical_scores = [round(dict_stereotypical_scores[w.lower()], 3) for w in context_list]
            bias_differences = [round(bias_diff, 3) for bias_diff in context["bias_differences"]]
            formatted_context_list = [f"\\hlrelated{{{w}}}" if bool_related else w for w, bool_related in zip(context_list, related_words)]
            formatted_context_list = [f"\\context{{{w}}}" for w in formatted_context_list]
            print(f"\n{context_list}\n{stereotypical_scores}\n{bias_differences}\n{related_words}")

            text_chains = [f"{w} \\& " for w in formatted_context_list]
            text_chain = ''.join(text_chains)[:-3] + "\\\\\n"

            score_chains = [f"\\textbf{{{stereo_score}}} \\& " for stereo_score in stereotypical_scores]
            score_chain = ''.join(score_chains)[:-3] + "\\\\\n"

            biasdiff_chains = [f"{biasdiff} \\& " for biasdiff in bias_differences]
            biasdiff_chain = ''.join(biasdiff_chains)[:-3] + "\\\\\n"
            tabout.write("\\begin{dependency}\n\\begin{deptext}[column sep=0.2cm]\n")
            tabout.write(text_chain)
            tabout.write(score_chain)
            tabout.write(biasdiff_chain)
            tabout.write("\\end{deptext}\n\\end{dependency} " +
                         f"& {round(100 * context['prob_he'], 1)} & {round(100 * context['prob_she'], 1)} & {round(100 * context['relevance_score'], 1)} & {round(context['bias_score'], 2)} \\\\ \\hline\n")
        else:
            print(f"\n{example} DOES NOT MEET OCCURRENCE THRESHOLD :\n"
                  f"{[dict_counts[word] for word in text_to_words(example).lower().split()]}")

    tabout.close()
