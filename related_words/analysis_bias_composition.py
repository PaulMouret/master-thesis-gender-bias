from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr, spearmanr
from collections import Counter
from blingfire import text_to_words
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import print_descriptive_statistics, print_full_descriptive_statistics, prettytable_to_latex, store_obj_in_jsonfile, load_obj_from_jsonfile, file_already_exists
from utils.constants_utils import *


# ! In order not to have biased counts, we lower the words to really have statistics for each word
# So we use .lower().split() instead of .split()

model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

corrupted_dataset_name = f'{full_dataset_name}_final_noising_experiment'

corrupted_dataset_original = load_dataset('json',
                                 data_files=f'../created_datasets/noising_experiment_datasets/'
                                            f'{corrupted_dataset_name}.jsonl',
                                 split='train')

#Analysis
print(f"We analyze {corrupted_dataset_name}\n")

# 1. We create the relevance boolean array dictionaries


dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                        f"dict_bias_diff_{subset_size}")
dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                     f"dict_counts_{subset_size}")
dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                   f"dict_stereotypical_scores_{subset_size}")

for type_dataset, name in zip(["all", "classical", "dialogue"
                               ],
                              ["", "classical_", "dialogue_"
                               ]
                              ):
    if type_dataset == "all":
        corrupted_dataset = corrupted_dataset_original
    elif type_dataset == "classical":
        corrupted_dataset = corrupted_dataset_original.filter(lambda x: not x["dialogue_context"])
    elif type_dataset == "dialogue":
        corrupted_dataset = corrupted_dataset_original.filter(lambda x: x["dialogue_context"])


    def min_counts_in_context(context):
        return min([dict_counts[word] for word in text_to_words(context["text"]).lower().split()])

    before_len = len(corrupted_dataset)
    corrupted_dataset = corrupted_dataset.filter(lambda x: min_counts_in_context(x) >= OCCURRENCE_THRESHOLD)
    after_len = len(corrupted_dataset)
    print(f"Keeping contexts with only frequent words represents {after_len} ({100 * round(after_len/before_len, 2)}%)")

    print(f"\n\n####### {type_dataset.upper()} #######")

    real_biases = np.array([context["bias_score"] for context in corrupted_dataset], dtype=float)
    for related_bool in (True, False):
        related_str = "related" if related_bool else "global"
        for aggregating_function, aggregating_name in zip((np.min, np.mean, np.max, np.sum),
                                                      ('min','mean', 'max', 'sum')):
            estimated_biases = []
            for context in corrupted_dataset:
                words = text_to_words(context['text']).lower().split()
                if related_bool:
                    words = np.array(words)[context['related_words']]
                words = [w for w in words if dict_counts[w] >= OCCURRENCE_THRESHOLD]
                if not words:
                    estimated_bias = 0
                else:
                    estimated_bias = aggregating_function([dict_stereotypical_scores[w] for w in words])
                estimated_biases.append(estimated_bias)
            #Now we have estimated_biases of same size as real_biases
            estimated_biases = np.array(estimated_biases, dtype=float)

            print(f"{related_str} {aggregating_name} : pearsonr {pearsonr(real_biases, estimated_biases)} "
                  f"spearmanr {spearmanr(real_biases, estimated_biases)}")

            estimated_biases = estimated_biases.reshape((-1, 1))
            regressor = LinearRegression().fit(estimated_biases, real_biases)
            linear_estimates = regressor.predict(estimated_biases)
            rsquare = regressor.score(estimated_biases, real_biases)
            mse = mean_squared_error(real_biases, linear_estimates)
            print(f"{related_str} {aggregating_name} : R^2 {round(rsquare, 2)} MSE {round(mse, 2)} ; "
                  f"coeffs {regressor.intercept_}  {regressor.coef_}")

