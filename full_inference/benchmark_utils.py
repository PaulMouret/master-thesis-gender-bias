import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr, spearmanr

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import print_full_descriptive_statistics
from utils.constants_utils import *


def full_inference_analysis(res_he, res_she, biases, relevances, lengths, inference_name,
                            save_pictures=True, return_dict=False):
    dict_res = dict()

    res_he = np.array(res_he)
    res_she = np.array(res_she)
    biases = np.array(biases)
    bias_measures = np.abs(biases)
    relevances = np.array(relevances)

    res_he *= 100 #because we will represent the percentages
    res_she *= 100
    relevances *= 100

    for vector, title, xlabel, size_bin,\
        low_range, high_range in zip([res_he, res_she, biases, bias_measures, relevances],
                                     ["res_he", "res_she", "biases", "bias_measures", "relevances"],
                                     ["Probability of 'he' (in %)",
                                      "Probability of 'she' (in %)",
                                      "Bias score",
                                      "Bias measure",
                                      "Relevance score"],
                                     [0.5, 0.5, 0.1, 0.1, 0.5],
                                     [0, 0, -6, 0, 0],
                                     [25, 25, 6, 6, 40]):

        if save_pictures:
            plt.hist(vector, bins=int((high_range - low_range)//size_bin), range=(low_range, high_range), density=True)
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.savefig(f"../saved_objects/benchmark/histogram_{title}_{inference_name}.png", dpi=200)
            plt.close()

        #Descriptive statistics
        new_res = print_full_descriptive_statistics(vector, xlabel, [0., 1., 5.], return_dict=return_dict)
        if return_dict:
            dict_res.update(new_res)

    for vector, title, y_label in zip([res_he, res_she, relevances],
                             ["res_he", "res_she", "relevances"],
                             ["Probability of 'he' (in %)",
                              "Probability of 'she' (in %)",
                              "Relevance score (in %)"]):
        vector = np.sort(vector)[::-1]

        if save_pictures:
            plt.bar(100 * np.arange(len(vector))/len(vector), vector)
            plt.ylabel(y_label)
            plt.xlabel("Proportion of contexts (in %)")
            plt.ylim(top=10)
            plt.savefig(f"../saved_objects/benchmark/bar_plot_{title}_{inference_name}.png", dpi=200)
            plt.close()

    #Is there a link between bias and relevance ?
    print(f"correlation coefficient between relevance and bias : "
          f"{pearsonr(relevances, biases)} ; {spearmanr(relevances, biases)}")

    print(f"correlation coefficient between relevance and bias measures : "
          f"{pearsonr(relevances, bias_measures)} ; {spearmanr(relevances, bias_measures)}")

    if save_pictures:
        plt.plot(relevances, biases, 'o', markersize=1)
        plt.xlim(right=40)
        plt.xlabel("Relevance score (in %)")
        plt.ylabel("Bias score")
        plt.savefig(f"../saved_objects/benchmark/bias_of_relevance_plot_{inference_name}.png", dpi=200)
        plt.close()

        plt.plot(relevances, biases, 'o', markersize=1)
        plt.xlim(left=0, right=2)
        plt.xlabel("Relevance score (in %)")
        plt.ylabel("Bias score")
        plt.savefig(f"../saved_objects/benchmark/bias_of_relevance_plot_{inference_name}_zoom.png", dpi=200)
        plt.close()

        plt.plot(relevances, bias_measures, 'o', markersize=1)
        plt.xlim(left=0, right=40)
        plt.xlabel("Relevance score (in %)")
        plt.ylabel("Bias measure")
        plt.savefig(f"../saved_objects/benchmark/bias_measure_of_relevance_plot_{inference_name}.png", dpi=200)
        plt.close()

    # Is there a link between length and relevance ?
    print(f"\ncorrelation coefficient between relevance and length : "
          f"{pearsonr(relevances, lengths)} ; {spearmanr(relevances, lengths)}")
    if save_pictures:
        plt.plot(lengths, relevances, 'o', markersize=1)
        plt.xlabel("Length of the context")
        plt.ylabel("Relevance score (in %)")
        plt.ylim(top=40)
        plt.xlim(right=30)
        plt.savefig(f"../saved_objects/benchmark/relevance_of_length_plot_{inference_name}.png", dpi=200)
        plt.close()

    # Is there a link between length and bias ?
    print(f"\ncorrelation coefficient between bias and length : "
          f"{pearsonr(biases, lengths)} ; {spearmanr(biases, lengths)}")

    print(f"\ncorrelation coefficient between bias measure and length : "
          f"{pearsonr(bias_measures, lengths)} ; {spearmanr(bias_measures, lengths)}")

    #Maybe we should work with proportion of relevant contexts instead of direct relevance score
    sorted_lengths = list(set(lengths))
    sorted_lengths.sort()
    relevance_proportions = []
    length_counts = []
    for l in sorted_lengths:
        l_indices = np.array(lengths) == l
        l_relevances = relevances[l_indices]
        l_relevance_proportion = np.count_nonzero(l_relevances >= RELEVANCE_THRESHOLD) / len(l_relevances)
        relevance_proportions.append(l_relevance_proportion)
        length_counts.append(len(l_relevances))
    relevance_proportions = 100 * np.array(relevance_proportions)

    print(f"\ncorrelation coefficient between relevance proportion and length :"
          f"{pearsonr(relevance_proportions, sorted_lengths)} ; {spearmanr(relevance_proportions, sorted_lengths)}")
    if save_pictures:
        plt.plot(sorted_lengths, relevance_proportions, 'o', markersize=1)
        plt.xlabel("Length of the context")
        plt.ylabel("Proportion of relevant contexts (in %)")
        plt.xlim(right=30)
        plt.savefig(f"../saved_objects/benchmark/relevance_prop_of_length_plot_{inference_name}.png", dpi=200)
        plt.close()

    #To have an idea of the number of contexts per length :
    if save_pictures:
        plt.plot(sorted_lengths, length_counts, 'o', markersize=1)
        plt.xlabel("Length of the context")
        plt.ylabel("Number of contexts")
        plt.xlim(right=30)
        plt.savefig(f"../saved_objects/benchmark/number_of_length_plot_{inference_name}.png", dpi=200)
        plt.close()

    return dict_res
