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

from utils.global_utils import print_descriptive_statistics, print_full_descriptive_statistics, prettytable_to_latex
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

for type_dataset in ["all", "classical", "dialogue"]:
    if type_dataset == "all":
        corrupted_dataset = corrupted_dataset_original
    elif type_dataset == "classical":
        corrupted_dataset = corrupted_dataset_original.filter(lambda x: not x["dialogue_context"])
    elif type_dataset == "dialogue":
        corrupted_dataset = corrupted_dataset_original.filter(lambda x: x["dialogue_context"])

    print(f"\n\n####### {type_dataset.upper()} #######")



    # I. Global vectors
    print("\n##############\nGlobal vectors (ie. as many observations as corrupted contexts)\n")

    # 1. Statistics and distribution
    print("\n##\nStatistics and distribution\n")

    # a. bias difference
    all_bias_differences = corrupted_dataset["bias_differences"] # a list of lists
    all_bias_differences = np.array([bias_diff for context in all_bias_differences for bias_diff in context]) # we flatten it
    #Statistics
    print_full_descriptive_statistics(all_bias_differences, "bias_differences", cdf_values=[0.])
    #Histogram
    plt.hist(all_bias_differences, bins=100, range=(-3, 3), density=True)
    plt.xlabel("Bias difference")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/histogram_bias_difference_{type_dataset}.png", dpi=200)
    plt.close()

    # b. corrupted bias
    all_corrupted_biases = corrupted_dataset["corrupted_biases"] # a list of lists
    all_corrupted_biases = np.array([bias for context in all_corrupted_biases for bias in context]) # we flatten it

    # c. original bias (but broadcasted to the shape of all_bias_differences)
    original_biases = np.array([context["bias_score"] for context in corrupted_dataset
                                for i in range(len(context["bias_differences"]))])

    # d. relevance difference
    all_relevance_differences = corrupted_dataset["relevance_differences"] # a list of lists
    all_relevance_differences = np.array([rel_diff for context in all_relevance_differences for rel_diff in context]) # we flatten it
    all_relevance_differences *= 100 #because we want percents
    #Statistics
    print_full_descriptive_statistics(all_relevance_differences, "relevance_differences")
    #Histogram
    plt.hist(all_relevance_differences, bins=100, range=(-40, 40), density=True)
    plt.xlabel("Relevance (in %) difference")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/histogram_relevance_difference_{type_dataset}.png", dpi=200)
    plt.close()

    # e. corrupted relevance
    all_corrupted_relevances = corrupted_dataset["corrupted_relevances"] # a list of lists
    all_corrupted_relevances = np.array([rel for context in all_corrupted_relevances for rel in context]) # we flatten it
    all_corrupted_relevances *= 100 #because we want %

    # f. original relevance (but broadcasted to the shape of all_relevance_differences)
    original_relevances = np.array([context["relevance_score"] for context in corrupted_dataset
                                    for i in range(len(context["bias_differences"]))])
    original_relevances *= 100 #because we want %

    # g. distance from the end
    distance_from_end = np.array([len(context)-1 - i for context in corrupted_dataset["corrupted_relevances"]
                                  for i in range(len(context))])


    # 2. Correlations and interactions
    print("\n##\nCorrelations and interactions\n")

    # a. Correlation with original scores

    print(f"correlation coefficient between original bias and bias difference :"
          f"{pearsonr(original_biases, all_bias_differences)} ; {spearmanr(original_biases, all_bias_differences)}")
    plt.plot(original_biases, all_bias_differences, 'o', markersize=1)
    plt.xlabel("Bias score")
    plt.ylabel("Bias difference")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/bias_diff_of_original_bias_{type_dataset}.png", dpi=200)
    plt.close()

    print(f"correlation coefficient between original bias and corrupted bias :"
          f"{pearsonr(original_biases, all_corrupted_biases)} ; {spearmanr(original_biases, all_corrupted_biases)}")
    plt.plot(original_biases, all_corrupted_biases, 'o', markersize=1)
    plt.xlabel("Bias score")
    plt.ylabel("Corrupted bias")
    #We draw the identity function :
    plt.plot([np.min(original_biases), np.max(original_biases)], [np.min(original_biases), np.max(original_biases)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/corrupted_bias_of_original_bias_{type_dataset}.png", dpi=200)
    plt.close()

    print(f"correlation coefficient between original relevance and relevance difference :"
          f"{pearsonr(original_relevances, all_relevance_differences)} ; "
          f"{spearmanr(original_relevances, all_relevance_differences)}")
    plt.plot(original_relevances, all_relevance_differences, 'o', markersize=1)
    plt.xlabel("Relevance score (in %)")
    plt.ylabel("Relevance (in %) difference")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/rel_diff_of_original_rel_{type_dataset}.png", dpi=200)
    plt.close()

    print(f"correlation coefficient between original relevance and corrupted relevance :"
          f"{pearsonr(original_relevances, all_corrupted_relevances)} ; "
          f"{spearmanr(original_relevances, all_corrupted_relevances)}")
    plt.plot(original_relevances, all_corrupted_relevances, 'o', markersize=1)
    plt.xlabel("Relevance score (in %)")
    plt.ylabel("Corrupted relevance score (in %)")
    #We draw the identity function :
    plt.plot([np.min(original_relevances), np.max(original_relevances)], [np.min(original_relevances), np.max(original_relevances)],
             color='r', linestyle='-', linewidth=1)
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/corrupted_rel_of_original_rel_{type_dataset}.png", dpi=200)
    plt.close()

    # b. Correlation between relevance difference and bias difference

    print(f"correlation coefficient between bias difference and relevance difference :"
          f"{pearsonr(all_bias_differences, all_relevance_differences)} ; "
          f"{spearmanr(all_bias_differences, all_relevance_differences)}")
    plt.plot(all_bias_differences, all_relevance_differences, 'o', markersize=1)
    plt.xlabel("Bias difference")
    plt.ylabel("Relevance (in %) difference")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/rel_diff_of_bias_diff_{type_dataset}.png", dpi=200)
    plt.close()

    #c . Correlation of rel diff and bias diff with distance from the end

    print(f"correlation coefficient between distance from end and relevance difference :"
          f"{pearsonr(distance_from_end, all_relevance_differences)} ; "
          f"{spearmanr(distance_from_end, all_relevance_differences)}")
    plt.plot(distance_from_end, all_relevance_differences, 'o', markersize=1)
    plt.xlabel("Distance from the end")
    plt.ylabel("Relevance (in %) difference")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/rel_diff_of_end_distance_{type_dataset}.png", dpi=200)
    plt.close()

    print(f"correlation coefficient between distance from end and bias difference :"
          f"{pearsonr(distance_from_end, all_bias_differences)} ; {spearmanr(distance_from_end, all_bias_differences)}")
    plt.plot(distance_from_end, all_bias_differences, 'o', markersize=1)
    plt.xlabel("Distance from the end")
    plt.ylabel("Bias difference")
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/bias_diff_of_end_distance_{type_dataset}.png", dpi=200)
    plt.close()


    # II. Contexts vectors (ie. of the same length as contexts)
    print("\n#############\nContexts vectors (ie. of the same length as contexts)\n")

    # 1. About the bias
    print("\n##\nAbout the bias\n")
    #Bias score
    contexts_biases = np.array(corrupted_dataset["bias_score"])
    #Naive estimation of global bias based on bias differences
    contexts_bias_diff_sum = np.array([np.sum(context["bias_differences"]) for context in corrupted_dataset])
    #Correlation between both
    print(f"correlation coefficient between contexts_biases and contexts_bias_diff_sum :"
          f"{pearsonr(contexts_biases, contexts_bias_diff_sum)} ; {spearmanr(contexts_biases, contexts_bias_diff_sum)}")
    plt.plot(contexts_biases, contexts_bias_diff_sum, 'o', markersize=1)
    plt.xlabel("Bias score")
    plt.ylabel("Sum of bias differences of words of the context")
    plt.plot([np.min(contexts_biases), np.max(contexts_biases)], [np.min(contexts_biases), np.max(contexts_biases)],
             color='r', linestyle='-', linewidth=1) #We draw the identity function
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/bias_diff_sum_of_bias_{type_dataset}.png", dpi=200)
    plt.close()

    # III. Focus on crucial words
    relevance_threshold = RELEVANCE_THRESHOLD
    NB_MOST_COMMON = 50
    print("\n###############\nFocus on crucial words\n")

    #Corrupted relevances
    corrupted_relevances = np.array(
            [cr for context_crs in corrupted_dataset["corrupted_relevances"] for cr in context_crs])

    #The following vectors have length nb_non_relevant (with duplicate)
    nr_indexes_from_beginning = [] #vector of indices
    nr_indexes_from_end = [] #vector of indices
    nr_words = [] #vector of words
    nr_base_relevances = [] #vector of relevances

    for context in corrupted_dataset:
        len_context = len(context["corrupted_relevances"])
        for i, cr in enumerate(context["corrupted_relevances"]):
            if cr < relevance_threshold:
                index_from_beginning = i
                index_from_end = len_context - 1 - i
                word = text_to_words(context['text']).lower().split()[i]

                nr_indexes_from_beginning.append(index_from_beginning)
                nr_indexes_from_end.append(index_from_end)
                nr_words.append(word)
                nr_base_relevances.append(context["relevance_score"])

                #print(f"{context['text']} # {text_to_words(context['text']).lower().split()[i]} ({i}) "
                      #f"# rel % : {100*cr} (from {100*context['relevance_score']})")

    nr_indexes_from_beginning = np.array(nr_indexes_from_beginning)
    nr_indexes_from_end = np.array(nr_indexes_from_end)
    counter_nr_indexes_from_beginning = Counter(nr_indexes_from_beginning)
    counter_nr_indexes_from_end = Counter(nr_indexes_from_end)

    #We construct frequencies from counts :
    nb_non_relevant = len(nr_indexes_from_beginning)
    for dictionary in (counter_nr_indexes_from_beginning, counter_nr_indexes_from_end):
        for k in dictionary:
            dictionary[k] = dictionary[k] / nb_non_relevant

    #We construct boolean vectors to manage overlapping indices ; same size as nr_words
    both_end_and_beginning = np.bitwise_and((nr_indexes_from_beginning == 0), (nr_indexes_from_end == 0))
    both_before_last_and_beginning = np.bitwise_and((nr_indexes_from_beginning == 0), (nr_indexes_from_end == 1))
    neither_end_nor_beginning = np.bitwise_and((nr_indexes_from_beginning != 0), (nr_indexes_from_end != 0))
    neither_end_nor_previous_nor_beginning = np.bitwise_and(neither_end_nor_beginning, (nr_indexes_from_end != 1))

    #We check the words at different indices
    nr_first_words = list(np.array(nr_words)[(nr_indexes_from_beginning == 0)])
    nr_end_words = list(np.array(nr_words)[(nr_indexes_from_end == 0)])
    nr_before_end_words = list(np.array(nr_words)[(nr_indexes_from_end == 1)])
    nr_middle_words = list(np.array(nr_words)[neither_end_nor_beginning])
    nr_real_middle_words = list(np.array(nr_words)[neither_end_nor_previous_nor_beginning])

    #We create counters
    nr_words_count = Counter(nr_words)
    nr_first_words_count = Counter(nr_first_words)
    nr_end_words_count = Counter(nr_end_words)
    nr_before_end_words_count = Counter(nr_before_end_words)
    nr_middle_words_count = Counter(nr_middle_words)
    nr_real_middle_words_count = Counter(nr_real_middle_words)
    #We convert counts into frequencies ; frequency compared to nb_non_relevant
    for dictionary in (nr_words_count, nr_first_words_count, nr_end_words_count, nr_before_end_words_count,
                       nr_middle_words_count, nr_real_middle_words_count):
        for k in dictionary:
            dictionary[k] = dictionary[k] / nb_non_relevant

    # 0. The amount of crucial = corrupted non relevant contexts
    print("\n##\nThe amount of crucial = corrupted non relevant contexts\n")
    relevant_proportion = np.count_nonzero(corrupted_relevances >= relevance_threshold) / len(corrupted_relevances)
    print(f"Amount of non-relevant corrupted contexts : {nb_non_relevant} ({nb_non_relevant/len(corrupted_relevances)})")
    print(f"relevant proportion : {relevant_proportion}")

    # 1. The index of crucial words
    print("\n##\nThe index of crucial words\n")

    #Distribution of indices of crucial words
    print("\nDistribution of indices")
    print("index of crucial words from beginning (proportion) : ",
          counter_nr_indexes_from_beginning.most_common(NB_MOST_COMMON))
    print("index of crucial words from end (proportion) : ", counter_nr_indexes_from_end.most_common(NB_MOST_COMMON))
    #Histogram of crucial words indices from the beginning
    plt.hist(nr_indexes_from_beginning, bins=np.max(nr_indexes_from_beginning)+1, density=True)
    plt.xlabel("Index (from the beginning) of crucial words")
    plt.ylabel("Frequency")
    plt.ylim(top=0.35)
    plt.xlim(right=40)
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/histogram_nr_indexes_from_beginning_{type_dataset}.png", dpi=200)
    plt.close()
    #Histogram of crucial words indices from the end
    plt.hist(nr_indexes_from_end, bins=np.max(nr_indexes_from_end)+1, density=True)
    plt.xlabel("Index (from the end) of crucial words")
    plt.ylabel("Frequency")
    plt.ylim(top=0.35)
    plt.xlim(right=40)
    plt.savefig(f"../saved_objects/noising_experiment/stats_final_experiment/histogram_nr_indexes_from_end_{type_dataset}.png", dpi=200)
    plt.close()

    #To what extent do indices overlap :
    print("\nTo what extent do indices overlap :")
    print("nb of contexts of length 1 originally :",
          corrupted_dataset.filter(lambda context: len(context["corrupted_relevances"])==1).shape[0])
    print("nb of contexts of length 2 originally :",
          corrupted_dataset.filter(lambda context: len(context["corrupted_relevances"])==2).shape[0])
    print("nb of crucial words being both end and beginning : ", np.count_nonzero(both_end_and_beginning))
    print("nb of crucial words being both before last and beginning : ", np.count_nonzero(both_before_last_and_beginning))
    print("nb of crucial words being neither at the end nor beginning : ", len(nr_middle_words))

    # 2. The type of crucial words
    print("\n##\nThe type of crucial words\n")

    #Most common crucial words, depending on indices
    print("\nMost common crucial words, depending on indices :")
    print("most common crucial words :", nr_words_count.most_common(NB_MOST_COMMON))
    print("most common first crucial words :", nr_first_words_count.most_common(NB_MOST_COMMON))
    print("most common end crucial words :", nr_end_words_count.most_common(NB_MOST_COMMON))
    print("most common before end crucial words :", nr_before_end_words_count.most_common(NB_MOST_COMMON))
    print("most common middle crucial words :", nr_middle_words_count.most_common(NB_MOST_COMMON))
    print("most common real middle crucial words :", nr_real_middle_words_count.most_common(NB_MOST_COMMON))

    #We construct the corresponding table:
    columns_names = ["word", "first", "middle", "penultimate", "last", "total"]
    sum_row = np.zeros(5)

    table = PrettyTable(columns_names)
    symbols = ('"', "'", ",", ".", "!", "?", "the", "i", "it", "was", "that", "and", "as", "to", "a")
    for symbol in symbols:
        first = round(100 * nr_first_words_count[symbol], 1)
        middle = round(100 * nr_real_middle_words_count[symbol], 1)
        penultimate = round(100 * nr_before_end_words_count[symbol], 1)
        last = round(100 * nr_end_words_count[symbol], 1)
        total = round(100 * nr_words_count[symbol], 1)
        row_values = [first, middle, penultimate, last, total]
        row = [symbol] + row_values
        table.add_row(row)
        sum_row += np.array(row_values)

    first_total = round(100 * nr_first_words_count.total(), 1)
    middle_total = round(100 * nr_real_middle_words_count.total(), 1)
    penultimate_total = round(100 * nr_before_end_words_count.total(), 1)
    last_total = round(100 * nr_end_words_count.total(), 1)
    total_total = round(100 * nr_words_count.total(), 1) #should be 100
    total_values = [first_total, middle_total, penultimate_total, last_total, total_total]
    table.add_row(["[other]"] + list(np.round(np.array(total_values) - sum_row, 1)))
    table.add_row(["total"] + total_values)
    # The table is filled
    # Now we can save it
    chain = table.get_string()
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/noising_experiment/stats_final_experiment"
    with open(f"{repertory}/crucial_words_{type_dataset}.txt", 'w') as f:
        f.write(chain)

    prettytable_to_latex(f"{repertory}/crucial_words_{type_dataset}")

    #To give a comparison :
    #Most common words, depending on indices

    #First we create corresponding objects with a similar code
    indexes_from_beginning = [i for corrupted_relevances in corrupted_dataset["corrupted_relevances"]
                              for i in range(len(corrupted_relevances))]
    indexes_from_end = [len(corrupted_relevances) - 1 - i
                        for corrupted_relevances in corrupted_dataset["corrupted_relevances"]
                        for i in range(len(corrupted_relevances))]
    words = [word for text in corrupted_dataset["text"] for word in text_to_words(text).lower().split()]
    #We create arrays and counters
    indexes_from_beginning = np.array(indexes_from_beginning)
    indexes_from_end = np.array(indexes_from_end)
    counter_indexes_from_beginning = Counter(indexes_from_beginning)
    counter_indexes_from_end = Counter(indexes_from_end)
    #We construct frequencies from counts :
    nb_words = len(words)
    for dictionary in (counter_indexes_from_beginning, counter_indexes_from_end):
        for k in dictionary:
            dictionary[k] = dictionary[k] / nb_words
    #We construct boolean vectors to manage overlapping indices ; same size as nb_words
    both_end_and_beginningw = np.bitwise_and((indexes_from_beginning == 0), (indexes_from_end == 0))
    both_before_last_and_beginningw = np.bitwise_and((indexes_from_beginning == 0), (indexes_from_end == 1))
    neither_end_nor_beginningw = np.bitwise_and((indexes_from_beginning != 0), (indexes_from_end != 0))
    neither_end_nor_previous_nor_beginningw = np.bitwise_and(neither_end_nor_beginningw, (indexes_from_end != 1))
    #We check the words at different indices
    first_words = list(np.array(words)[(indexes_from_beginning == 0)])
    second_words = list(np.array(words)[(indexes_from_beginning == 1)])
    end_words = list(np.array(words)[(indexes_from_end == 0)])
    before_end_words = list(np.array(words)[(indexes_from_end == 1)])
    middle_words = list(np.array(words)[neither_end_nor_beginningw])
    real_middle_words = list(np.array(words)[neither_end_nor_previous_nor_beginningw])
    #We create counters
    words_count = Counter(words)
    first_words_count = Counter(first_words)
    second_words_count = Counter(second_words)
    end_words_count = Counter(end_words)
    before_end_words_count = Counter(before_end_words)
    middle_words_count = Counter(middle_words)
    real_middle_words_count = Counter(real_middle_words)
    #We convert counts into frequencies ; frequency compared to nb_non_relevant
    for dictionary in (words_count, first_words_count, end_words_count, before_end_words_count,
                       middle_words_count, real_middle_words_count, second_words_count):
        for k in dictionary:
            dictionary[k] = dictionary[k] / nb_words
    #We print the results
    print("\nMost common words, depending on indices :")
    print(f"(number of words, with duplicate : {nb_words})")
    print("most common words :", words_count.most_common(NB_MOST_COMMON))
    print("most common first words :", first_words_count.most_common(NB_MOST_COMMON))
    print("most common end words :", end_words_count.most_common(NB_MOST_COMMON))
    print("most common before end words :", before_end_words_count.most_common(NB_MOST_COMMON))
    print("most common middle words :", middle_words_count.most_common(NB_MOST_COMMON))
    print("most common real middle words :", real_middle_words_count.most_common(NB_MOST_COMMON))

    # We construct the corresponding table:
    columns_names = ["word", "first", "middle", "penultimate", "last", "total"]
    sum_row = np.zeros(5)

    table = PrettyTable(columns_names)
    symbols = ('"', "'", ",", ".", "!", "?", "the", "i", "it", "was", "that", "and", "as", "to", "a")
    for symbol in symbols:
        first = round(100 * first_words_count[symbol], 1)
        middle = round(100 * real_middle_words_count[symbol], 1)
        penultimate = round(100 * before_end_words_count[symbol], 1)
        last = round(100 * end_words_count[symbol], 1)
        total = round(100 * words_count[symbol], 1)
        row_values = [first, middle, penultimate, last, total]
        row = [symbol] + row_values
        table.add_row(row)
        sum_row += np.array(row_values)

    first_total = round(100 * first_words_count.total(), 1)
    middle_total = round(100 * real_middle_words_count.total(), 1)
    penultimate_total = round(100 * before_end_words_count.total(), 1)
    last_total = round(100 * end_words_count.total(), 1)
    total_total = round(100 * words_count.total(), 1)  # should be 100
    total_values = [first_total, middle_total, penultimate_total, last_total, total_total]
    table.add_row(["[other]"] + list(np.round(np.array(total_values) - sum_row, 1)))
    table.add_row(["total"] + total_values)
    # The table is filled
    # Now we can save it
    chain = table.get_string()
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/noising_experiment/stats_final_experiment"
    with open(f"{repertory}/common_words_{type_dataset}.txt", 'w') as f:
        f.write(chain)

    prettytable_to_latex(f"{repertory}/common_words_{type_dataset}")

    if type_dataset in ("all", "dialogue"):

        # 3. The link between cruciality and dialogue
        print("\n##\nThe link between cruciality and dialogue\n")

        #Cruciality of quotation marks
        print("\nCruciality of quotation marks :")
        dialogue_contexts = corrupted_dataset.filter(lambda context: context['dialogue_context'])
        print(f"Amount of dialogue contexts : {len(dialogue_contexts)} ({len(dialogue_contexts)/len(corrupted_dataset)})")
        nr_first_dialogue_contexts = dialogue_contexts.filter(lambda context: context["corrupted_relevances"][0] < relevance_threshold)
        nr_end_dialogue_contexts = dialogue_contexts.filter(lambda context: context["corrupted_relevances"][-1] < relevance_threshold)
        first_dialogue_french = dialogue_contexts.filter(lambda context: context['text'][0] == '"')
        print("proportion of dialogue contexts where first quote crucial : ",
              len(nr_first_dialogue_contexts)/len(dialogue_contexts))
        print("proportion of dialogue contexts where end quote crucial : ",
              len(nr_end_dialogue_contexts)/len(dialogue_contexts))
        print('proportion of dialogue contexts where the first quote is " : ',
              len(first_dialogue_french) / len(dialogue_contexts))

        #Cruciality of before last word
        print("\nCruciality of before last word")
        nr_punctuation_before_last = corrupted_dataset.filter(lambda context:
                                                              (len(context["corrupted_relevances"]) > 1 and
                                                               text_to_words(context['text']).lower().split()[-2] in [".", "?", "!"]) and
                                                              context["corrupted_relevances"][-2] < relevance_threshold)
        dialogue_contexts_with_punctuation = dialogue_contexts.filter\
            (lambda context: len(context["corrupted_relevances"]) > 1 and text_to_words(context["text"]).lower().split()[-2] in [".", "?", "!"])
        nr_dialogue_contexts_with_punctuation = dialogue_contexts_with_punctuation.filter\
            (lambda context: context["corrupted_relevances"][-2] < relevance_threshold)
        print(f"proportion of dialogue contexts with punctuation : "
              f"{len(dialogue_contexts_with_punctuation)/len(dialogue_contexts)}")
        print(f"proportion of dialogue contexts with punctuation where punctuation crucial : "
              f"{len(nr_dialogue_contexts_with_punctuation)/len(dialogue_contexts_with_punctuation)}")
        print("\n")


    # 4. The link between cruciality and base relevance
    print("\n##\nThe link between cruciality and base relevance\n")
    print("Here are the statistics of base relevance scores depending on the index of crucial words :")
    base_relevances = np.array(corrupted_dataset["relevance_score"])
    nr_base_relevances = np.array(nr_base_relevances)
    #We check the words at different indices
    nr_first_words_base_relevances = nr_base_relevances[(nr_indexes_from_beginning == 0)]
    nr_end_words_base_relevances = nr_base_relevances[(nr_indexes_from_end == 0)]
    nr_before_end_words_base_relevances = nr_base_relevances[(nr_indexes_from_end == 1)]
    nr_middle_words_base_relevances = nr_base_relevances[neither_end_nor_beginning]
    nr_real_middle_words_base_relevances = nr_base_relevances[neither_end_nor_previous_nor_beginning]
    for relevance_vector, relevance_vector_name in zip([base_relevances, nr_base_relevances, nr_first_words_base_relevances,
                                                        nr_end_words_base_relevances, nr_before_end_words_base_relevances,
                                                        nr_middle_words_base_relevances,
                                                        nr_real_middle_words_base_relevances],
                                                       ['base_relevances', 'nr_base_relevances', 'nr_first_words_base_relevances',
                                                        'nr_end_words_base_relevances', 'nr_before_end_words_base_relevances',
                                                        'nr_middle_words_base_relevances',
                                                        'nr_real_middle_words_base_relevances']
                                                       ):
        print_descriptive_statistics(relevance_vector, relevance_vector_name)

    #We search for example of contexts
    example_dataset1 = corrupted_dataset.filter(lambda x: len(x['corrupted_relevances']) > 3 and
                                                          x['corrupted_relevances'][0] < relevance_threshold and
                                                          x['corrupted_relevances'][-1] < relevance_threshold and
                                                          x['corrupted_relevances'][-2] < relevance_threshold)
    print(f"For (dialogue ?) examples:")
    for i in range(5):
        print(f"{example_dataset1['text'][i]} : {text_to_words(example_dataset1['text'][i]).split()} "
              f"({np.array(example_dataset1['corrupted_relevances'][i]) < relevance_threshold})")
    if type_dataset != "dialogue":
        example_dataset2 = corrupted_dataset.filter(lambda x: np.count_nonzero(np.array(x['corrupted_relevances']) < relevance_threshold) >= 2
                                                              and not x["dialogue_context"])
        print(f"For classical examples:")
        for i in range(10):
            print(f"{example_dataset1['text'][i]} : {text_to_words(example_dataset1['text'][i]).split()} "
                  f"({np.array(example_dataset1['corrupted_relevances'][i]) < relevance_threshold})")


    #We also print the median bias difference for each index, even if don't expect special results
    NB_MOST_COMMON = 50
    print("\n###############\nFocus on bias differences\n")

    all_bias_differences = corrupted_dataset["bias_differences"]  # a list of lists
    all_indexes_from_beginning = np.array(
        [i for context in all_bias_differences for i in range(len(context))])  # we flatten it
    all_indexes_from_end = np.array(
        [len(context) - 1 - i for context in all_bias_differences for i in range(len(context))])  # we flatten it
    all_bias_differences = np.array(
        [bias_diff for context in all_bias_differences for bias_diff in context])  # we flatten it

    all_words = []
    #It is possible to define it by comprehension by there would be too many calls to text_to_words than needed
    for context in corrupted_dataset:
        context_list = text_to_words(context['text']).lower().split()
        len_context = len(context_list)
        for i, word in enumerate(context_list):
            all_words.append(word)
    all_words = np.array(all_words)

    print(f"We check the length is the same : {len(all_bias_differences)}, {len(all_indexes_from_beginning)}, "
          f"{len(all_indexes_from_end)}, {len(all_words)}")

    counter_all_indexes_from_beginning = Counter(all_indexes_from_beginning)
    counter_all_indexes_from_end = Counter(all_indexes_from_end)

    # We construct boolean vectors to manage overlapping indices ; same size as nr_words
    all_both_end_and_beginning = np.bitwise_and((all_indexes_from_beginning == 0), (all_indexes_from_end == 0))
    all_both_before_last_and_beginning = np.bitwise_and((all_indexes_from_beginning == 0), (all_indexes_from_end == 1))
    all_neither_end_nor_beginning = np.bitwise_and((all_indexes_from_beginning != 0), (all_indexes_from_end != 0))
    all_neither_end_nor_previous_nor_beginning = np.bitwise_and(all_neither_end_nor_beginning, (all_indexes_from_end != 1))

    # We check the words at different indices
    all_first_words = list(np.array(nr_words)[(nr_indexes_from_beginning == 0)])
    all_end_words = list(np.array(nr_words)[(nr_indexes_from_end == 0)])
    all_before_end_words = list(np.array(nr_words)[(nr_indexes_from_end == 1)])
    all_middle_words = list(np.array(nr_words)[neither_end_nor_beginning])
    all_real_middle_words = list(np.array(nr_words)[neither_end_nor_previous_nor_beginning])

    print("\n##\nThe index of bias differences\n")

    data = [np.median(all_bias_differences[all_indexes_from_beginning == i]) for i in range(40)]
    print(f"meidan bias differences starting from the beginning:\n{data}")
    fig, ax = plt.subplots()
    ax.bar(np.arange(40), data)
    plt.xlabel("Index (from the beginning)")
    plt.ylabel("Median bias difference")
    plt.ylim(top=0.5, bottom=-0.5)
    plt.xlim(right=20)
    plt.savefig(
        f"../saved_objects/noising_experiment/stats_final_experiment/barplot_bias_diff_from_beginning_{type_dataset}.png",
        dpi=200)
    plt.close()

    data = [np.median(all_bias_differences[all_indexes_from_end == i]) for i in range(40)]
    fig, ax = plt.subplots()
    ax.bar(np.arange(40), data)
    plt.xlabel("Index (from the end)")
    plt.ylabel("Median bias difference")
    plt.ylim(top=0.5, bottom=-0.5)
    plt.xlim(right=20)
    plt.savefig(
        f"../saved_objects/noising_experiment/stats_final_experiment/barplot_bias_diff_from_end_{type_dataset}.png",
        dpi=200)
    plt.close()

    #We reprint the most common words at given positions
    print("\nWe print most common words, depending on indices :")
    print("most common words :", words_count.most_common(NB_MOST_COMMON))
    print("most common first words :", first_words_count.most_common(NB_MOST_COMMON))
    print("most common second words :", second_words_count.most_common(NB_MOST_COMMON))


    #For the two first indexes we print
    for important_index in (0, 1):
        print(f"\nimportant index {important_index}")
        symbols = ('"', "'", ",", ".", "!", "?", "the", "i", "it", "was", "that", "and", "as", "to", "a")
        for word in symbols:
            word_bias_differences_on_index = all_bias_differences[np.bitwise_and(all_words == word,
                                                                                 all_indexes_from_beginning == important_index)
                                                                  ]
            word_bias_differences = all_bias_differences[all_words == word]
            mean_bias_diffs_on_index = round(np.mean(word_bias_differences_on_index), 3)
            mean_bias_diffs = round(np.mean(word_bias_differences), 3)
            print(f"{word} : mean bias diff at {important_index} {mean_bias_diffs_on_index} ; "
                  f"mean bias diff {mean_bias_diffs}")

    #For the plot
    NB_FREQUENT = 200
    for index, freq_dict in zip([0,1], [first_words_count, second_words_count]):
        index_str = "first" if index == 0 else "second"
        list_mean_bias_diffs_on_index = []
        list_mean_bias_diffs = []
        sum_freq = 0
        list_freq = []
        for word, freq in freq_dict.most_common(NB_FREQUENT):
            freq /= freq_dict.total()
            #original freq is the frequence relative to all words ; we want it relative to the index
            sum_freq += freq
            list_freq.append(sum_freq)
            word_bias_differences_on_index = all_bias_differences[np.bitwise_and(all_words == word,
                                                                                 all_indexes_from_beginning == index)
            ]
            word_bias_differences = all_bias_differences[all_words == word]
            list_mean_bias_diffs_on_index.append(np.mean(word_bias_differences_on_index))
            list_mean_bias_diffs.append(np.mean(word_bias_differences))

        print(f"The {NB_FREQUENT} most frequent words at index {index} in {type_dataset} represent "
              f"{round(100 * sum_freq, 2)}% of occurrences at this index")

        relative_changes = (np.array(list_mean_bias_diffs) - np.array(list_mean_bias_diffs_on_index)) / np.array(list_mean_bias_diffs)
        print(f"{type_dataset} dataset, index {index} :")
        print(f"Average relative difference : {100 * np.mean(relative_changes)} %")
        print(f"Median relative difference : {100 * np.median(relative_changes)} %")

        plt.plot(list_mean_bias_diffs, list_mean_bias_diffs_on_index, 'o', markersize=3)
        plt.xlabel("Stereotypical score over all indices")
        plt.ylabel(f"Stereotypical score over {index_str} index")
        # We draw the identity function :
        plt.plot([np.min(list_mean_bias_diffs), np.max(list_mean_bias_diffs)],
                 [np.min(list_mean_bias_diffs), np.max(list_mean_bias_diffs)],
                 color='r', linestyle='-', linewidth=1)
        plt.savefig(
            f"../saved_objects/noising_experiment/stats_final_experiment/{index}_ss_of_all_ss_{type_dataset}.png",
            dpi=200)
        plt.close()

        #To know what NB_FREQUENT to choose
        plt.plot(100 * np.array(list_freq), 'o', markersize=2)
        plt.xlabel("Number of most common words")
        plt.ylabel(f"Cumulative frequency at index {index} (in %)")
        plt.xlim(right=200)
        plt.savefig(
            f"../saved_objects/noising_experiment/stats_final_experiment/cumulative_freq_{index}_{type_dataset}.png",
            dpi=200)
        plt.close()
