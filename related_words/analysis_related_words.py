from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr, spearmanr
from collections import Counter
from blingfire import text_to_words
from prettytable import PrettyTable

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

    dataset = corrupted_dataset #because I adapt code from several sources, enable to be consistent

    # 1. We create the relevance boolean array dictionaries
    if not file_already_exists("../saved_objects/noising_experiment/stereotypical_scores/",
                               f"{name}dict_related_{subset_size}.json"):
        dict_related = {}
        for ic, context in enumerate(dataset):
            for i, w in enumerate(text_to_words(context["text"]).lower().split()):
                if w not in dict_related.keys():
                    dict_related[w] = [context["related_words"][i]]
                else:
                    dict_related[w].append(context["related_words"][i])
        store_obj_in_jsonfile(dict_related, "../saved_objects/noising_experiment/related_words/",
                              f"{name}dict_related_{subset_size}")
    else:
        dict_related = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                              f"{name}dict_related_{subset_size}")
        #The elements of this dict are matching with the elemtns of the corresponding dict_bias_diff

    dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                            f"{name}dict_bias_diff_{subset_size}")
    dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                         f"{name}dict_counts_{subset_size}")
    dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                       f"{name}dict_stereotypical_scores_{subset_size}")

    dict_related_counts = {word: np.count_nonzero(dict_related[word]) for word in dict_related.keys()}
    dict_related_stereotypical_scores = {word: np.mean(np.array(dict_bias_diff[word])[np.array(dict_related[word])])
                                          for word in dict_bias_diff.keys()}
    store_obj_in_jsonfile(dict_counts, "../saved_objects/noising_experiment/related_words/",
                          f"{name}dict_related_counts_{subset_size}")
    store_obj_in_jsonfile(dict_stereotypical_scores, "../saved_objects/noising_experiment/related_words/",
                          f"{name}dict_related_stereotypical_scores_{subset_size}")

    dict_unrelated_counts = {word: np.count_nonzero(np.bitwise_not(dict_related[word])) for word in dict_related.keys()}
    dict_unrelated_stereotypical_scores = {word: np.mean(np.array(dict_bias_diff[word])[np.bitwise_not(dict_related[word])])
                                         for word in dict_bias_diff.keys()}
    store_obj_in_jsonfile(dict_counts, "../saved_objects/noising_experiment/related_words/",
                          f"{name}dict_unrelated_counts_{subset_size}")
    store_obj_in_jsonfile(dict_stereotypical_scores, "../saved_objects/noising_experiment/related_words/",
                          f"{name}dict_unrelated_stereotypical_scores_{subset_size}")

    print(f"\n\n####### {type_dataset.upper()} #######")

    related_words = np.array(
            [rw for context_rws in corrupted_dataset["related_words"] for rw in context_rws])
    prop_related = np.count_nonzero(related_words) / len(related_words)
    print(f"Proportion of related words : {round(100 * prop_related, 2)} % "
          f"(over {len(related_words)} words, with duplicate)")

    frequent_related_and_unrelated = [w for w in dict_counts.keys() if dict_related_counts[w] >= OCCURRENCE_THRESHOLD
                                      and dict_unrelated_counts[w] >= OCCURRENCE_THRESHOLD]
    nb_frequent_related_and_unrelated = len(frequent_related_and_unrelated)
    print(f"There are {nb_frequent_related_and_unrelated} "
          f"({round(100 * nb_frequent_related_and_unrelated / len(dict_counts), 2)}%) "
          f"of words whose related and unrelated occurrences meet the occurrence threshold {OCCURRENCE_THRESHOLD}")

    frequent_related_scores = [dict_related_stereotypical_scores[w] for w in frequent_related_and_unrelated]
    frequent_unrelated_scores = [dict_unrelated_stereotypical_scores[w] for w in frequent_related_and_unrelated]
    global_scores = [dict_stereotypical_scores[w] for w in frequent_related_and_unrelated]
    frequent_related_scores = np.array(frequent_related_scores)
    frequent_unrelated_scores = np.array(frequent_unrelated_scores)
    global_scores = np.array(global_scores)
    order = np.argsort(global_scores)

    plt.plot(global_scores, frequent_related_scores, 'o', markersize=1)
    plt.xlabel("Global stereotypical score")
    plt.ylabel("Related stereotypical score")
    plt.xlim(right=2.)
    # We draw the identity function :
    plt.plot([np.min(global_scores), np.max(global_scores)],
             [np.min(global_scores), np.max(global_scores)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/related_words/related_of_global_{type_dataset}.png", dpi=200)
    plt.close()

    plt.plot(global_scores, frequent_unrelated_scores, 'o', markersize=2, color='k', label='unrelated scores')
    plt.plot(global_scores, frequent_related_scores, 'o', markersize=2, color='tab:blue', label='related scores')
    plt.xlim(right=2.)
    plt.xlabel("Global stereotypical score")
    plt.ylabel("Related and unrelated stereotypical score")
    plt.legend()
    # We draw the identity function :
    plt.plot([np.min(global_scores), np.max(global_scores)],
             [np.min(global_scores), np.max(global_scores)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/related_words/related_unrelated_of_global_{type_dataset}.png", dpi=200)
    plt.close()

    plt.plot(global_scores, frequent_unrelated_scores, 'o', markersize=1)
    plt.xlabel("Global stereotypical score")
    plt.ylabel("Unrelated stereotypical score")
    # We draw the identity function :
    plt.plot([np.min(global_scores), np.max(global_scores)],
             [np.min(global_scores), np.max(global_scores)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/related_words/unrelated_of_global_{type_dataset}.png", dpi=200)
    plt.close()

    plt.plot(frequent_unrelated_scores, frequent_related_scores, 'o', markersize=1)
    plt.xlabel("Global stereotypical score")
    plt.ylabel("Related stereotypical score")
    # We draw the identity function :
    plt.plot([np.min(frequent_unrelated_scores), np.max(frequent_unrelated_scores)],
             [np.min(frequent_unrelated_scores), np.max(frequent_unrelated_scores)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/related_words/related_of_unrelated_{type_dataset}.png", dpi=200)
    plt.close()

    all_bias_differences = np.array([bias_diff for context in corrupted_dataset["bias_differences"] for bias_diff in context])
    related_bias_differences = all_bias_differences[related_words]
    unrelated_bias_differences = all_bias_differences[np.bitwise_not(related_words)]

    #We compare the two distributions
    # Statistics
    print_full_descriptive_statistics(related_bias_differences, "related_bias_differences", cdf_values=[0.])
    # Histogram
    plt.hist(related_bias_differences, bins=100, range=(-3, 3), density=True)
    plt.xlabel("Related bias difference")
    plt.ylabel("Density")
    plt.savefig(
        f"../saved_objects/noising_experiment/related_words/histogram_related_bias_difference_{type_dataset}.png",
        dpi=200)
    plt.close()

    # Statistics
    print_full_descriptive_statistics(unrelated_bias_differences, "unrelated_bias_differences", cdf_values=[0.])
    # Histogram
    plt.hist(unrelated_bias_differences, bins=100, range=(-3, 3), density=True)
    plt.xlabel("Unrelated bias difference")
    plt.ylabel("Density")
    plt.savefig(
        f"../saved_objects/noising_experiment/related_words/histogram_unrelated_bias_difference_{type_dataset}.png",
        dpi=200)
    plt.close()

    #Focus on person-hyponyms
    print(f"\n#######\nFocus on person-hyponyms")
    clean_gender_of_persons = load_obj_from_jsonfile('../created_datasets/utils_datasets/', "clean_gender_of_persons")
    neutral_persons = [person['word'] for person in clean_gender_of_persons if person['gender'] == 'n' and person['word'] in dict_related.keys()]
    bool_related_person_hyponyms = [boolean for w in neutral_persons for boolean in dict_related[w] for w in neutral_persons]
    print(f"Proportion of related person-hyponyms: "
          f"{np.count_nonzero(bool_related_person_hyponyms) / len(bool_related_person_hyponyms)} "
          f"({np.count_nonzero(bool_related_person_hyponyms)})")

    # 2. Stereotypical scores
    print("\n\n###########\n2. Stereotypical scores")
    frequency_threshold = OCCURRENCE_THRESHOLD
    NB_MOST_COMMON = 400

    # Words with most stereotypes (ie. largest and lowest stereotypical scores)
    counter_male_stereotype = Counter(dict_stereotypical_scores)
    counter_female_stereotype = Counter({word: -(dict_stereotypical_scores[word])
                                         for word in dict_stereotypical_scores.keys()})
    # the dict does not really contain counts, but the method still returns the keys with highest values

    # Words with most stereotypes for the chosen frequency threshold, and their score
    counter_male_stereotype_with_threshold = Counter({word: dict_stereotypical_scores[word]
                                                      for word in dict_stereotypical_scores.keys()
                                                      if dict_counts[word] >= frequency_threshold})
    counter_female_stereotype_with_threshold = Counter({word: -(dict_stereotypical_scores[word])
                                                        for word in dict_stereotypical_scores.keys()
                                                        if dict_counts[word] >= frequency_threshold})
    print("\nMale stereotypes :")
    for word, score in counter_male_stereotype_with_threshold.most_common(NB_MOST_COMMON):
        if dict_counts[word] >= frequency_threshold and dict_related_counts[word] >= frequency_threshold and dict_unrelated_counts[word] >= frequency_threshold:
            print(f"{word} : {round(score, 2)} ({dict_counts[word]}  occurences) ; "
                  f"{round(dict_related_stereotypical_scores[word], 2)} "
                  f"({dict_related_counts[word]} related occurences) ;"
                  f"{round(dict_unrelated_stereotypical_scores[word], 2)} "
                  f"({dict_unrelated_counts[word]} unrelated occurences)"
                  )
    print("\nFemale stereotypes :")
    for word, score in counter_female_stereotype_with_threshold.most_common(NB_MOST_COMMON):
        if dict_counts[word] >= frequency_threshold and dict_related_counts[word] >= frequency_threshold and dict_unrelated_counts[word] >= frequency_threshold:
            print(f"{word} : {round(score, 2)} ({dict_counts[word]}  occurences) ; "
                  f"{round(- dict_related_stereotypical_scores[word], 2)} "
                  f"({dict_related_counts[word]} related occurences) ;"
                  f"{round(- dict_unrelated_stereotypical_scores[word], 2)} "
                  f"({dict_unrelated_counts[word]} unrelated occurences)"
                  )
