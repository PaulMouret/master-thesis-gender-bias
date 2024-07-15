import hashlib
import re
from datasets import load_dataset
from blingfire import text_to_words
import json
import numpy as np

import sys

from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile


#In order to create the dataset (of contexts) and its statistics, and several utils datasets and statistics around it


def generate_unfiltered_contexts_from_jsonl_sentences(jsonl_name, dataset_name, min_len=1, verbose=True):
    """
    Looks for contexts, ie. beginning of sentences, whose next word is likely to be a pronoun ;
    given a list of sentences, this correspond to the beginning of sentences that are actually followed by a pronoun
    in the data.
    It generates a jsonl document for the dataset, and several json documents for the statistics

    IMPORTANT : for the function to have the expected behavior, the sentences in the json file should:
    - correspond to actual sentences (by opposition to pieces of sentences)

    Parameters
    -----------
    jsonl_name: str
        the name of the jsonlfile (including its path, and without its extension), that contains the data.
        each element of the dataset is a dictionary with the key 'text', whose value corresponds to a sentence
    dataset_name: str
        will be used to name the generated dataset
    min_len: int
        the minimum length of contexts
    verbose: bool
        indicates if the program prints some information while running

    Returns
    ___________
    contexts_name: str
        the name of the created dataset (without the extension of the file)
    """
    # Because these are large datasets, we won't store several variations of the datasets
    # (like person-neutral contexts, etc.) just to get the corresponding statistics :
    # we will rather compute the different statistics at the same time we create the contexts

    contexts_name = "unfiltered_contexts_" + dataset_name

    pronouns = {"he", "she"} #originally there were also "they", "i", "we", but it could give weird contexts
    gendered_pronouns = {"he", "she"}

    one_character_quotation_marks = ["\u201c", "\u201d", "\""]
    two_character_quotation_marks = ["''", "``"]
    #In practice, using \u201c and \u201d instead slightly improves relevance ; plus it enables some uniformization
    #But at the end, I prefere not to do such an uniformization

    seen = set()
    #To remove duplicates. Note that, for better memory usage, it will contains hashtable of contexts
    #instead of contexts themselves

    #We look at two types of contexts : classical and dialogue

    with open(f'../source_datasets/{jsonl_name}.jsonl') as f:
        with open(f'../created_datasets/{contexts_name}.jsonl', 'w') as contexts:
            for line in f:
                sentence = json.loads(line)['text']

                #classical context
                gendered_pronoun_encountered = False
                # to cut the going through the sentence if 'he' or 'she' previsouly encountered
                tokenized_sentence = text_to_words(sentence)
                for i_word, word in enumerate(tokenized_sentence.split()):
                    if not gendered_pronoun_encountered and (word in pronouns) and (i_word >= min_len):
                        if word in gendered_pronouns:
                            gendered_pronoun_encountered = True
                        # 1. We have to reconstruct the context, because there is no obvious correspondence between
                        # the indices of sentence and text_to_words(sentence) ; and words may be repeating
                        # Following operations look costly, but it's not them that influence running time the most
                        tokenized_context = ' '.join(tokenized_sentence.split()[:i_word])
                        id_in_tokenized_sentence = len(tokenized_context) + 1
                        occurences_in_tokenized_sentence = [m.start() for m in re.finditer(word, tokenized_sentence)]
                        nb_occurence = occurences_in_tokenized_sentence.index(id_in_tokenized_sentence)
                        id_in_sentence = [m.start() for m in re.finditer(word, sentence)][nb_occurence]
                        context = sentence[:id_in_sentence].strip() #because there may be a remaining space at the end
                        #real context done

                        context_hash = hashlib.md5(context.encode()).digest()
                        classical_dialogue = (context[:2] in two_character_quotation_marks and context[-2:] in two_character_quotation_marks) \
                                             or (context[:1] in one_character_quotation_marks and context[-1:] in one_character_quotation_marks)
                        # because some dialogues are included inside one sentence with their dialogue attribution
                        if context_hash not in seen:
                            seen.add(context_hash)
                            contexts.write(json.dumps({'text': context, 'dialogue_context': classical_dialogue,
                                                       'original_text': sentence}) + "\n")

                #dialogue context
                #For dialogue, quotation marks may be representend in different ways,
                #and sometimes even in an inconsitent way
                #(between the right side and the left side), so this has to be taken into account


                dialogue = (sentence[:2] in two_character_quotation_marks or sentence[:1] in one_character_quotation_marks) \
                           and (sentence[-2:] in two_character_quotation_marks or sentence[-1:] in one_character_quotation_marks)
                if dialogue:
                    context = sentence
                    context_hash = hashlib.md5(context.encode()).digest()
                    if context_hash not in seen:
                        seen.add(context_hash)
                        contexts.write(json.dumps({'text': context, 'dialogue_context': True,
                                                   'original_text': sentence + " he said."}) + "\n")
                        #By default I use 'he' pronoun, because the coreference solver seems male-biased.
                        # ex. : The doctor/nurse had a long day ahead of him/her.

    print(f'Created ../created_datasets/{contexts_name}.jsonl')


def filter_dataset(dataset_name):
    my_dataset = load_dataset('json', data_files=f'../created_datasets/unfiltered_contexts_{dataset_name}.jsonl',
                              split='train')
    my_dataset = my_dataset.shuffle(seed=42) #to avoid having all sentences from the same book in a row
    # 1. We make it person-neutral
    gendered_persons = load_obj_from_jsonfile("../created_datasets/utils_datasets/", "gendered_persons")  # a list
    manual_gendered_persons = load_obj_from_jsonfile("../created_datasets/utils_datasets/",
                                                     "manual_gendered_persons")  # a list
    gendered_persons += manual_gendered_persons
    gendered_persons = set(gendered_persons)
    #Using a set is supposed to be better, but in practive it doesn't change much

    person_neutral_dataset = my_dataset.filter(
        lambda x: not any({word in gendered_persons
                           for word in text_to_words(x['text'].lower()).split()}))
    person_neutral_dataset = person_neutral_dataset.filter(
        lambda x: not any({expression in text_to_words(x['text'].lower())
                           for expression in ['lounge lizard' 'sex kitten', 'sumo wrestler', 'wet nurse']}))

    #I only consider a few factually gendered expressions that consist in several words
    #(because almost all of them are composed of one word that is factually gendered) ;
    #if I change my mind, see my remarks in gender_of_persons.py
    print("Filtered factually gendered persons")

    # 2. We make it firstname-neutral
    gendered_firstnames = load_obj_from_jsonfile("../created_datasets/utils_datasets/", "gendered_firstnames")  # a list
    manual_gendered_firstnames = load_obj_from_jsonfile("../created_datasets/utils_datasets/",
                                                        "manual_gendered_firstnames")  # a list
    gendered_firstnames += manual_gendered_firstnames
    gendered_firstnames = set(gendered_firstnames)
    # gendered_firstnames are first names that are English or of length > 3
    # If I only want English first names, use "eng_gendered_firstnames" instead

    person_and_firstname_neutral_dataset = person_neutral_dataset.filter(
        lambda x: not any({word in gendered_firstnames for word in text_to_words(x['text']).split()}))
    print("Filtered factually gendered first names")

    person_and_firstname_neutral_dataset.to_json(f'../created_datasets/contexts_{dataset_name}.jsonl')

    # 3. We compute the statistics
    #Doing it now enables not to save the intermediate dataset person_neutral_dataset
    max_len = 50

    #Not the cleaner code, but we simply do it for the six subsets we are interested in

    # a. unfiltered
    classical_unfiltered = my_dataset.filter(lambda x: not x['dialogue_context'])
    classical_unfiltered_lengths = [len(text_to_words(example['text']).split()) for example in classical_unfiltered]
    hist_classical_unfiltered, bin_edges = np.histogram(classical_unfiltered_lengths, bins=max_len+1, range=(0, max_len+1))
    nb_classical_unfiltered = len(classical_unfiltered)

    dialogue_unfiltered = my_dataset.filter(lambda x: x['dialogue_context'])
    dialogue_unfiltered_lengths = [len(text_to_words(example['text']).split()) for example in dialogue_unfiltered]
    hist_dialogue_unfiltered, bin_edges = np.histogram(dialogue_unfiltered_lengths, bins=max_len + 1,
                                                       range=(0, max_len + 1))
    nb_dialogue_unfiltered = len(dialogue_unfiltered)

    # b. person_neutral
    classical_person_neutral = person_neutral_dataset.filter(lambda x: not x['dialogue_context'])
    classical_person_neutral_lengths = [len(text_to_words(example['text']).split()) for example in classical_person_neutral]
    hist_classical_person_neutral, bin_edges = np.histogram(classical_person_neutral_lengths, bins=max_len + 1,
                                                        range=(0, max_len + 1))
    nb_classical_person_neutral = len(classical_person_neutral)

    dialogue_person_neutral = person_neutral_dataset.filter(lambda x: x['dialogue_context'])
    dialogue_person_neutral_lengths = [len(text_to_words(example['text']).split()) for example in dialogue_person_neutral]
    hist_dialogue_person_neutral, bin_edges = np.histogram(dialogue_person_neutral_lengths, bins=max_len + 1,
                                                       range=(0, max_len + 1))
    nb_dialogue_person_neutral = len(dialogue_person_neutral)

    # c. final = person_neutral + firstname_neutral
    classical_final = person_and_firstname_neutral_dataset.filter(lambda x: not x['dialogue_context'])
    classical_final_lengths = [len(text_to_words(example['text']).split()) for example in classical_final]
    hist_classical_final, bin_edges = np.histogram(classical_final_lengths, bins=max_len + 1,
                                                        range=(0, max_len + 1))
    nb_classical_final = len(classical_final)

    dialogue_final = person_and_firstname_neutral_dataset.filter(lambda x: x['dialogue_context'])
    dialogue_final_lengths = [len(text_to_words(example['text']).split()) for example in dialogue_final]
    hist_dialogue_final, bin_edges = np.histogram(dialogue_final_lengths, bins=max_len + 1,
                                                       range=(0, max_len + 1))
    nb_dialogue_final = len(dialogue_final)

    #From there, we build the arrays
    c_nb_contexts = np.array([nb_classical_unfiltered, nb_classical_person_neutral, nb_classical_final], dtype=int)
    d_nb_contexts = np.array([nb_dialogue_unfiltered, nb_dialogue_person_neutral, nb_dialogue_final], dtype=int)

    c_length_contexts = np.array([hist_classical_unfiltered, hist_classical_person_neutral, hist_classical_final],
                                 dtype=int)
    d_length_contexts = np.array([hist_dialogue_unfiltered, hist_dialogue_person_neutral, hist_dialogue_final],
                                 dtype=int)

    # We can reconstruct contexts statistics (for all contexts) by summing classical and dialogue statistics
    nb_contexts = c_nb_contexts + d_nb_contexts
    length_contexts = c_length_contexts + d_length_contexts

    classical_contexts_statistics = {'nb_contexts': c_nb_contexts.tolist(),
                                     'length_contexts': c_length_contexts.tolist()}
    dialogue_contexts_statistics = {'nb_contexts': d_nb_contexts.tolist(),
                                    'length_contexts': d_length_contexts.tolist()}
    contexts_statistics = {'nb_contexts': nb_contexts.tolist(),
                           'length_contexts': length_contexts.tolist()}

    store_obj_in_jsonfile(classical_contexts_statistics, '../saved_objects/context_dataset_creation/'
                                                         'created_statistics/',
                          f"statistics_classical_contexts_{dataset_name}")
    store_obj_in_jsonfile(dialogue_contexts_statistics, '../saved_objects/context_dataset_creation/'
                                                        'created_statistics/',
                          f"statistics_dialogue_contexts_{dataset_name}")
    store_obj_in_jsonfile(contexts_statistics, '../saved_objects/context_dataset_creation/created_statistics/',
                          f"statistics_contexts_{dataset_name}")

    print("Computed statistics")


def generate_contexts_from_jsonl_sentences(jsonl_name, dataset_name, min_len=1, verbose=True):
    generate_unfiltered_contexts_from_jsonl_sentences(jsonl_name, dataset_name, min_len=min_len, verbose=verbose)
    filter_dataset(dataset_name)
