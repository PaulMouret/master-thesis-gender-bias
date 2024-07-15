import torch
import numpy as np
import matplotlib.pyplot as plt

import sys

from utils.inference_utils import load_model, load_tokenizer, get_pronouns_indices, get_pronouns_probabilities
from utils.global_utils import print_descriptive_statistics, print_full_descriptive_statistics


# RUNNING INFERENCE

def get_padded_strings_pronoun_probablities(strings, nb_padding_range, model, tokenizer, pronouns, verbose=False):
    """
    Returns the pronouns probablities for a set of input strings, for all possible numbers of padding tokens
    below nb_padding_range

    Parameters
    -----------
    strings: List[str]
        a set of input strings
    nb_padding_range: int
        indicates how many padding sequences we test for each input string
    model:
        the model we want to use
    tokenizer:
        the tokenizer we want to use
        (it should correspond to the model)
    pronouns: List[str]
        a list of pronouns
        Note that, depending on the model, we may want to add a space " " before
    verbose: bool
        indicates the verbosity of the function

    Returns
    ___________
    res: list
        a list containing, for each input sentence, for each padding length to test, the probabilities of the
        pronouns
    """
    res = []
    pronouns_indices = get_pronouns_indices(pronouns, tokenizer)

    #tokenization of the strings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #We will tokenize them one by one

    model.eval()
    with torch.no_grad():

        for id_string, string in enumerate(strings):
            res.append([])
            input_id_len = len(tokenizer.encode(string, return_tensors="pt", padding=False)[0])
            for i in range(nb_padding_range):
                new_input_id = tokenizer(string, return_tensors="pt", padding='max_length',
                                         max_length=input_id_len+i, return_token_type_ids=False).to(device)
                if verbose and id_string == 0:
                    print(f"{i} padding characters :")
                    print(new_input_id["input_ids"])
                    print(new_input_id["attention_mask"])
                logits = model(**new_input_id).logits
                pron_probs = get_pronouns_probabilities(logits, pronouns_indices)[0].tolist()
                res[-1].append(pron_probs)

    return res


def get_llama_padded_strings_pronoun_probablities(strings, nb_padding_range, pad_token, verbose=False):
    """
    Returns, for the llama model, the pronouns probablities for a set of input strings,
    for all possible numbers of padding tokens below nb_padding_range

    Parameters
    -----------
    strings: List[str]
        a set of input strings
    nb_padding_range: int
        indicates how many padding sequences we test for each input string
    pad_token: str
        Indicates according to which policy the padding token should be defined
        choose among "bos", "eos", "pad"
    verbose: bool
        indicates the verbosity of the function

    Returns
    ___________
    res: list
        a list containing, for each input sentence, for each padding length to test, the probabilities of 'he' and 'she'
    """
    model_name = "meta-llama/Llama-2-7b-hf"
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name, pad_token=pad_token)
    pronouns = ['he', 'she']
    return get_padded_strings_pronoun_probablities(strings, nb_padding_range, model, tokenizer, pronouns, verbose)



# ANALYZING RESULTS

BIN_WIDTH = 1
POSITIVE_MAX_RANGE = 50


def log_bias(probs_he, probs_she):
    return np.log2(probs_he / probs_she)


def abs_log_bias(probs_he, probs_she):
    return np.abs(np.log2(probs_he / probs_she))


def sum_relevance(probs_he, probs_she):
    return np.add(probs_he, probs_she)


def relative_max_diff_function(vector):
    min_indices = np.argmin(vector, axis=1)
    max_indices = np.argmax(vector, axis=1)

    min_elements = vector[np.arange(vector.shape[0]), min_indices]
    if np.any(min_elements == 0):
        min_elements += np.finfo(np.float32).eps #we add "machine epsilon"
    max_elements = vector[np.arange(vector.shape[0]), max_indices]
    relative_max_diff = (max_elements - min_elements) / min_elements

    return relative_max_diff, min_indices, max_indices


def simple_max_diff_function(vector):
    min_indices = np.argmin(vector, axis=1)
    max_indices = np.argmax(vector, axis=1)

    min_elements = vector[np.arange(vector.shape[0]), min_indices]
    max_elements = vector[np.arange(vector.shape[0]), max_indices]
    simple_max_diff = max_elements - min_elements
    return simple_max_diff, min_indices, max_indices


def pronoun_probabilities_analysis(res_pronoun):
    return relative_max_diff_function(res_pronoun)


def measure_analysis(res_he, res_she, measure_function):
    #the measure function can be either a bias function or a relevance function
    measures = measure_function(res_he, res_she)
    #correct_measures = measures[:, 0]
    relative_max_diff, min_indices, max_indices = relative_max_diff_function(measures)
    simple_max_diff, min_indices, max_indices = simple_max_diff_function(measures)
    return relative_max_diff, simple_max_diff


def res_analysis(res, bias_function, relevance_function, dataset_name, addon_name=""):
    res = np.array(res)
    nb_sentences, padding_range, nb_pronouns = res.shape
    res_he = res[:, :, 0].squeeze()
    res_she = res[:, :, 1].squeeze()
    # res_he and res_she have shape (nb_sentences, padding_range)

    relative_max_diff_he, id_min_he, id_max_he = pronoun_probabilities_analysis(res_he)
    relative_max_diff_she, id_min_she, id_max_she = pronoun_probabilities_analysis(res_she)
    relative_max_diff_he *= 100 #because we will represent the percentages
    relative_max_diff_she *= 100

    # Indices of minimal and maximal probabilities
    for vector, title, xlabel in zip([id_min_he, id_max_he, id_min_she, id_max_she],
                                     ["id_min_he", "id_max_he", "id_min_she", "id_max_she"],
                                     ["Length of padding sequence with lowest probability for 'he'",
                                      "Length of padding sequence with largest probability for 'he'",
                                      "Length of padding sequence with lowest probability for 'she'",
                                      "Length of padding sequence with largest probability for 'she'"]):
        plt.hist(vector, bins=padding_range, range=(0, padding_range), density=True)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.savefig(f"../saved_objects/padding_analysis/histogram_{title}_{addon_name}_{dataset_name}.png", dpi=200)
        plt.close()

    #Relative maximum probability difference
    for vector, title, xlabel in zip([relative_max_diff_he, relative_max_diff_she],
                                     ["relative_max_diff_he", "relative_max_diff_she"],
                                     ["Relative maximum probability difference for 'he' (in %)",
                                      "Relative maximum probability difference for 'she' (in %)"]):
        plt.hist(vector, bins=POSITIVE_MAX_RANGE//BIN_WIDTH, range=(0, 10), density=True)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.savefig(f"../saved_objects/padding_analysis/histogram_{title}_{addon_name}_{dataset_name}.png", dpi=200)
        plt.close()

        print_descriptive_statistics(vector, title, cdf_values=[10])

    # Bias analysis
    relative_bias_difference, simple_bias_difference = measure_analysis(res_he, res_she, bias_function)
    relative_bias_difference *= 100
    plt.hist(relative_bias_difference, bins=POSITIVE_MAX_RANGE//BIN_WIDTH,
             range=(0, POSITIVE_MAX_RANGE), density=True)
    plt.xlabel("Relative maximum bias difference (in %)")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/padding_analysis/histogram_relative_bias_difference_{addon_name}_{dataset_name}.png",
                dpi=200)
    plt.close()
    print_descriptive_statistics(relative_bias_difference, "relative_bias_difference", cdf_values=[10])

    plt.hist(simple_bias_difference, bins=100,
             range=(0, 0.1), density=True)
    plt.xlabel("Maximum bias difference")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/padding_analysis/histogram_simple_bias_difference_{addon_name}_{dataset_name}.png",
                dpi=200)
    plt.close()
    print_full_descriptive_statistics(simple_bias_difference, "simple_bias_difference", cdf_values=[10])

    # Relevance analysis
    relative_relevance_difference, simple_relevance_difference = measure_analysis(res_he, res_she, relevance_function)
    relative_relevance_difference *= 100
    plt.hist(relative_relevance_difference, bins=POSITIVE_MAX_RANGE//BIN_WIDTH,
             range=(0, 10), density=True)
    plt.xlabel("Relative maximum relevance difference (in %)")
    plt.ylabel("Density")
    plt.savefig(f"../saved_objects/padding_analysis/histogram_relative_relevance_difference_{addon_name}_{dataset_name}.png",
                dpi=200)
    plt.close()
    print_descriptive_statistics(relative_relevance_difference, "relative_relevance_difference")
