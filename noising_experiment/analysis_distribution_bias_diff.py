from datasets import load_dataset
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr, kstest, mannwhitneyu, kruskal, t, normaltest
from statistics import stdev #the corrected standard deviation
from collections import Counter
from blingfire import text_to_words

import sys

from utils.global_utils import store_obj_in_jsonfile, load_obj_from_jsonfile, print_full_descriptive_statistics, print_descriptive_statistics, file_already_exists
from utils.constants_utils import *


# ! In order not to have biased counts, we lower the words to really have statistics for each word
# So we use .lower().split() instead of .split()

# I - Creation of useful objects

# 0. We load the dataset

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

dialogue_dataset = corrupted_dataset.filter(lambda context: context['dialogue_context'])
classical_dataset = corrupted_dataset.filter(lambda context: not context['dialogue_context'])

print(f"\nWe analyze {corrupted_dataset_name}.")

# 1 . Dictionary of bias differences
for dataset, name in zip([corrupted_dataset, dialogue_dataset, classical_dataset], ["", "dialogue_", "classical_"]):
    if not file_already_exists("../saved_objects/noising_experiment/stereotypical_scores/",
                               f"{name}dict_bias_diff_{subset_size}.json"):
        dict_bias_diff = {}
        for ic, context in enumerate(dataset):
            for i, w in enumerate(text_to_words(context["text"]).lower().split()):
                if w not in dict_bias_diff.keys():
                    dict_bias_diff[w] = [context["bias_differences"][i]]
                else:
                    dict_bias_diff[w].append(context["bias_differences"][i])
    else:
        dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                f"{name}dict_bias_diff_{subset_size}")

    # 2. Dictionary of counts
    dict_counts = {word: len(dict_bias_diff[word]) for word in dict_bias_diff.keys()}

    # 3. Dictionary of stereotypical scores
    dict_stereotypical_scores = {word: np.mean(dict_bias_diff[word]) for word in dict_bias_diff.keys()}

    # Now we save them all
    store_obj_in_jsonfile(dict_bias_diff, "../saved_objects/noising_experiment/stereotypical_scores/",
                          f"{name}dict_bias_diff_{subset_size}")
    store_obj_in_jsonfile(dict_counts, "../saved_objects/noising_experiment/stereotypical_scores/",
                          f"{name}dict_counts_{subset_size}")
    store_obj_in_jsonfile(dict_stereotypical_scores, "../saved_objects/noising_experiment/stereotypical_scores/",
                          f"{name}dict_stereotypical_scores_{subset_size}")

for dataset, name in zip([corrupted_dataset, dialogue_dataset, classical_dataset], ["", "dialogue_", "classical_"]):
    print(f"\n###### DATASET {name}")

    #Because dict variables have taken different values, we load them to be sure we have the expected ones
    dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                            f"{name}dict_bias_diff_{subset_size}")
    dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                         f"{name}dict_counts_{subset_size}")
    dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                       f"{name}dict_stereotypical_scores_{subset_size}")

    nb_words = len(dict_counts)
    print(f"There are {nb_words} different words (without duplicate).")

    # II - Analysis of the distribution of bias diff
    print("\n##\nAnalysis of the distribution of bias diff\n")
    ALPHA_VALUE = 0.05

    # 1. We test if bias_diffs (of a given word) follow a normal distribution
    #kstest is the Kolmogorov-Smirnov test, to evaluate goodness of fit.
    #We will evaluate the goodness of fit to a normal distribution.
    normal_test_K = 2 #must be >= 2 to avoid zero standard deviation
    dict_bias_diff_arrays = {word: np.array(bias_diffs) for word, bias_diffs in dict_bias_diff.items()
                             if dict_counts[word] >= normal_test_K}
    nb_normal_test = len(dict_bias_diff_arrays)
    print("\n##\nWe test if bias_diffs (of a given word) follow a normal distribution.\n")
    dict_normal_distrib = {word: kstest(rvs=(X - X.mean()) / (X.std()), cdf="norm", alternative="two-sided")[1]
                           for word, X in dict_bias_diff_arrays.items()}
    #The dictionary contains p-values
    nb_normal = sum(1 for pvalue in dict_normal_distrib.values() if pvalue >= ALPHA_VALUE)
    print(f"With significance level {ALPHA_VALUE}, {nb_normal} ({nb_normal/nb_normal_test}) of words "
          f"(that appear at least {normal_test_K} times) have a bias difference"
          f" that follows a normal distribution.")

    #We check if this proportion changes with the number of occurrences of the word
    list_prop_normal = []

    #We have to procede differently for words that appear once:
    #First we check that the test is always positive for 1 value:
    print(f"test for one value : {kstest(rvs=[0.], cdf='norm', alternative='two-sided')}")
    #Now it is sure, we can compute the frequency
    nb_one_occurrence = sum(1 for word, count in dict_counts.items() if count == 1)
    print(f"nb_one_occurrence : {nb_one_occurrence}")
    print(f"nb_normal_test : {nb_normal_test}")
    print(f" nb_normal_test + nb_one_occurrence ({nb_normal_test + nb_one_occurrence}) "
          f"should be equal to len(dict_bias_diff) ({len(dict_bias_diff)})")
    print(f"With significance level {ALPHA_VALUE}, {nb_normal + nb_one_occurrence} "
          f"({(nb_normal + nb_one_occurrence)/(nb_normal_test + nb_one_occurrence)}) of words "
          f"have a bias difference that follows a normal distribution.")
    list_prop_normal.append((nb_normal + nb_one_occurrence)/(nb_normal_test + nb_one_occurrence))

    print("\nWe check if this proportion changes with the number of occurrences of the word")
    for k in range(2, 30):
        dict_normal_distrib_k = {word: pval for word, pval in dict_normal_distrib.items() if dict_counts[word] >= k}
        nb_k = len(dict_normal_distrib_k)
        nb_normal_k = sum(1 for pvalue in dict_normal_distrib_k.values() if pvalue >= ALPHA_VALUE)
        print(f"With significance level {ALPHA_VALUE}, {nb_normal_k} ({nb_normal_k/nb_k}) of words "
              f"(that appear at least {k} times) have a bias difference that follows a normal distribution.")
        list_prop_normal.append(nb_normal_k/nb_k)

    plt.plot(list_prop_normal, marker='.')
    plt.xlim(right=20)
    plt.xlabel("Occurrence threshold")
    plt.ylabel("Proportion of normal distributions")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}prop_normal_distrib.png",
                dpi=200)
    plt.close()

    #It would be interesting to know if there is an interpretation of words which have normal distribution
    print("\nIt would be interesting to know if there is an interepretation of words which have normal distribution")
    words = list(dict_normal_distrib.keys())
    p_values_norm = list(dict_normal_distrib.values())
    order = np.argsort(p_values_norm) #in ascending order
    ordered_p_values = np.array(p_values_norm)[order]
    ordered_words = list(np.array(words)[order])
    ordered_counts = [dict_counts[word] for word in ordered_words]
    ordered_stereotypical_scores = [dict_stereotypical_scores[word] for word in ordered_words]
    #print(f"Words sorted by ascending p-value :\n{ordered_words}")
    #print(f"Counts sorted by ascending p-value :\n{ordered_counts}")
    #print(f"Stereotypical scores sorted by ascending p-value :\n{ordered_stereotypical_scores}")
    #We don't print them for readibility, but counts definitely play a role (the higher, the less likely to be normal)
    #Corresponding plots
    plt.plot(ordered_counts, ordered_p_values, 'o', markersize=1)
    plt.xlabel("Number of occurence of the word")
    plt.ylabel("p-value for normality test")
    plt.xlim(right=1000)
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}p_value_of_counts.png",
                dpi=200)
    plt.close()
    #Corresponding plots
    plt.plot(ordered_stereotypical_scores, ordered_p_values, 'o', markersize=1)
    plt.xlabel("Stereotypical of the word")
    plt.ylabel("p-value for normality test")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}p_value_of_stereotypical_score.png",
                dpi=200)
    plt.close()

    # 2. We test for which value of k we get reliable results
    print("\n##\nWe test for which value of k we get reliable results\n")
    MAX_TESTED_K = 20

    #Now we analyze the average width of confidence intervals
    print("\nNow we analyze the average width of confidence intervals\n")
    np.random.seed(42)
    avg_width_of_interval = []
    for k in range(2, MAX_TESTED_K + 1):
        frequent_dict_bias_diff_k = {word: bias_diffs for word, bias_diffs in dict_bias_diff.items()
                                     if dict_counts[word] >= k}
        nb_frequent_words_k = len(frequent_dict_bias_diff_k)

        quantile = t(df=k -1).ppf((ALPHA_VALUE/2, 1 - ALPHA_VALUE/2))[1]
        widthes = []

        for word, bias_diffs in frequent_dict_bias_diff_k.items():
            subset = random.sample(bias_diffs, k)
            s = stdev(subset)
            width = 2 *quantile * s / np.sqrt(k)
            widthes.append(width)

        avg_width = np.mean(widthes)
        avg_width_of_interval.append(avg_width)
        print(f"{k} : {avg_width} average width of interval")

    plt.plot(np.arange(2, MAX_TESTED_K + 1), avg_width_of_interval, marker='.')
    plt.xlabel("K")
    plt.xticks(np.array([2, 5, 10, 15, 20]))
    plt.ylabel("Average width of the 95%-confidence interval for s(w)")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}avg_width_of_k.png",
                dpi=200)
    plt.close()

    #Besides the interval width, let's print out the results for each (interesting) K
    NB_MOST_COMMON = 50
    #Words with most stereotypes, given some threshold
    print("\nWords with most stereotypes (with some frequency threshold) :")
    #Note that frequency is different from term frequency
    #To give an order of magnitude, for 10 000 contexts, there are 100 000 words
    occurrence_thresholds = np.arange(5, 12)
    for occurrence_threshold in occurrence_thresholds:
        print(f"Threshold {occurrence_threshold}")
        counter_male_stereotype_with_threshold = Counter({word: dict_stereotypical_scores[word]
                                                          for word in dict_stereotypical_scores.keys()
                                                          if dict_counts[word] >= occurrence_threshold})
        counter_female_stereotype_with_threshold = Counter({word: -(dict_stereotypical_scores[word])
                                                            for word in dict_stereotypical_scores.keys()
                                                            if dict_counts[word] >= occurrence_threshold})
        print("Male stereotypes :\n", counter_male_stereotype_with_threshold.most_common(NB_MOST_COMMON))
        print("Female stereotypes (in abs. value)\n: ",
              counter_female_stereotype_with_threshold.most_common(NB_MOST_COMMON))

    #We print out the distribution for all stereotypical scores, and for those that meet the threshold
    frequent_stereotypical_scores = np.array([dict_stereotypical_scores[word] for word, count in dict_counts.items()
                                              if count >= OCCURRENCE_THRESHOLD])
    all_stereotypical_scores = np.array([dict_stereotypical_scores[word] for word, count in dict_counts.items()])
    #Descriptive statistics
    print_full_descriptive_statistics(all_stereotypical_scores, "all_stereotypical_scores")
    print_full_descriptive_statistics(frequent_stereotypical_scores, "frequent_stereotypical_scores")
    #Histograms
    high_range = 2
    low_range = -2
    size_bin = 0.1
    #histogram all
    plt.hist(all_stereotypical_scores, bins=int((high_range - low_range)//size_bin), range=(low_range, high_range), density=True) #bins=int((high_range - low_range)//size_bin), range=(low_range, high_range)
    plt.xlabel("Stereotypical scores (without any occurrence threshold)")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/"
                f"{name}histogram_all_stereotypical_scores.png", dpi=200)
    plt.close()
    #histogram frequent
    plt.hist(frequent_stereotypical_scores, bins=int((high_range - low_range)//size_bin), range=(low_range, high_range), density=True) #bins=int((high_range - low_range)//size_bin), range=(low_range, high_range)
    plt.xlabel(f"Stereotypical scores (with occurrence threshold {OCCURRENCE_THRESHOLD})")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/"
                f"{name}histogram_frequent_stereotypical_scores.png", dpi=200)
    plt.close()
    #Now we check if the stereotypical scores are normal (it particular, it guarantees some symmetry)
    print("\nNow we check if the stereotypical scores are normal (it particular, it guarantees some symmetry)")
    pvalue_all = kstest(rvs=(all_stereotypical_scores - all_stereotypical_scores.mean()) / (all_stereotypical_scores.std()),
                        cdf="norm", alternative="two-sided")[1]
    pvalue_frequent = kstest(rvs=(frequent_stereotypical_scores - frequent_stereotypical_scores.mean()) / (frequent_stereotypical_scores.std()),
                        cdf="norm", alternative="two-sided")[1]

    print(f"With significance level {ALPHA_VALUE},"
          f"\nall_stereotyical_scores has a normal distrib : {pvalue_all >= ALPHA_VALUE}"
          f"\nfrequent_stereotyical_scores has a normal distrib : {pvalue_frequent >= ALPHA_VALUE}")


    # 3. We print the distribution of bias diff for some words
    print("\n##\nWe print the distribution of bias diff for some words\n")
    words_of_interest = []
    for word in words_of_interest:
        if word.isalpha(): #to make sure we can save the png files
            bias_diffs = dict_bias_diff[word]
            stereotypical_score = dict_stereotypical_scores[word]

            plt.hist(bias_diffs, bins=30, range=(-1.5, 1.5), density=True)
            plt.xlabel(f"Bias differences of {word}")
            plt.ylabel("Density")
            plt.vlines(stereotypical_score, 0, 1, colors='r')
            plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}histogram_bias_diff_{word}.png",
                        dpi=200)
            plt.close()


    # III - The distribution of counts and related statistics
    print("\n##############\nThe distribution of counts and related statistics\n")
    counts = list(dict_counts.values())
    print(f"Reminder : There are {nb_words} different words in the considered dataset,"
          f" which contains {len(dataset)} contexts.")

    # 1. Distribution of counts
    print("\n##\nDistribution of counts\n")
    #Descriptive statistics
    print_descriptive_statistics(counts, "counts")
    #histogram
    plt.hist(counts, range=(1, 21), bins=20, density=True) #bins=int((high_range - low_range)//size_bin), range=(low_range, high_range)
    plt.xlabel("Number of occurrences")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}histogram_counts.png", dpi=200)
    plt.close()

    list_at_least = []
    #As histogram is not very readible, we print the proportion of words that have at least k occurences
    print("\nAs histogram is not very readible, we print the proportion of words that have at least k occurences :")
    for k in range(1, MAX_TESTED_K + 1):
        frequent_dict_bias_diff_k = {word: bias_diffs for word, bias_diffs in dict_bias_diff.items()
                                     if dict_counts[word] >= k}
        dict_bias_diff_k = {word: bias_diffs for word, bias_diffs in dict_bias_diff.items()
                                     if dict_counts[word] == k}
        nb_frequent_words_k = len(frequent_dict_bias_diff_k)
        nb_words_k = len(dict_bias_diff_k)
        print(f"{nb_frequent_words_k} ({nb_frequent_words_k / nb_words}) have at least {k} occurences ; "
              f"{nb_words_k} ({nb_words_k / nb_words}) have at exactly {k} occurences")
        list_at_least.append(100 * (nb_frequent_words_k / nb_words))

    plt.plot(np.arange(1, MAX_TESTED_K + 1), list_at_least, marker='.')
    plt.xlabel("K")
    plt.ylabel("Proportion of frequent words (in %)")
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}prop_frequent_words.png",
                dpi=200)
    plt.close()

    #To have the intuition of which words have k occurrences
    print("\nTo have the intuition of which words have k occurrences :")
    for k in range(1, MAX_TESTED_K + 1):
        frequent_dict_bias_diff_equals_k = {word: bias_diffs for word, bias_diffs in dict_bias_diff.items()
                                            if dict_counts[word] == k}
        nb_frequent_words_k = len(frequent_dict_bias_diff_equals_k)
        print(f"{nb_frequent_words_k} ({nb_frequent_words_k / nb_words}) have exactly {k} occurences : "
              f"{list(frequent_dict_bias_diff_equals_k.keys())[:50]}")
    #To set a threshold
    vector = np.sort(counts)[::-1]
    plt.bar(100 * np.arange(len(vector))/len(vector), vector)
    plt.ylabel("Number of occurences")
    plt.xlabel("Proportion of frequent words (in %)")
    plt.ylim(top=40)
    plt.savefig(f"../saved_objects/noising_experiment/distribution_bias_diff/{name}bar_plot_counts.png", dpi=200)
    plt.close()

    # 2. The proportion of contexts that consist only in words that have at least k occurrences
    # Note sure that it turns out useful at the end
    print("\nThe proportion of contexts that consist only in words that have at least k occurrences :")
    def min_occurrence_in_context(context):
        return min([dict_counts[word] for word in text_to_words(context["text"]).lower().split()])
    #for k in range(1, MAX_TESTED_K):
        #frequent_dataset = dataset.filter(lambda x: min_occurrence_in_context(x) >= k)
        #print(f"There are {len(frequent_dataset)} ({len(frequent_dataset)/len(dataset)}) contexts"
              #f" that consist only in words with at least {k} occurrences.")
