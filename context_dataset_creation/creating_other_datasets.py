from datasets import load_dataset

import sys

from utils.global_utils import file_already_exists
from dataset_creation_utils import generate_contexts_from_jsonl_sentences


#We create a context dataset ; this file is devoted to smaller/example datasets

#We will use load_dataset(), so we will obtain Dataset instances
#In particular dataset['text'] is a list of strings
#These strings correspond to different things depending on datasets, so they will be described for each of them


#BOOKCORPUSOPEN
#similar to BookCorpus, except that it should be better regarding exposed defaults
#bookcorpusopen = load_dataset('bookcorpusopen', split='train')
#shows error
# FileNotFoundError: Couldn't find file at https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz


#SFGRAM
#Constituted of several text files (one file per book)
#a string doesn't correspond to a sentence and it is not preprocessed (regarding lowercase and sepration of punctuation)
#sf_gram_path = "source_datasets/SF-Gram-book-contents/"
#for book in get_list_of_filenames_from_path(sf_gram_path):
    #book_dataset = load_dataset('text', data_files=sf_gram_path+book, split='train')


#KAGGLE FICTION STORIES DATASET
#a string doesn't correspond to a sentence and it is not preprocessed (regarding lowercase and sepration of punctuation)
#It contains 29 401 strings
#jsonl_name = "kaggle_fiction"
#dataset_name = "kaggle_fiction"

#if not file_already_exists("../source_datasets/", f"{jsonl_name}.jsonl"):
    #kaggle_fiction_dataset = load_dataset('text', data_files='../source_datasets/kaggle_fiction_dataset.txt', split='train')
    #kaggle_fiction_dataset.to_json(f"../source_datasets/{jsonl_name}.jsonl")

#generate_contexts_from_jsonl_sentences(jsonl_name, dataset_name)
#in the future, for instance to put them in pipelines, the created dataset can be opened as a dataset object :
#my_dataset = load_dataset('json', data_files=f'created_datasets/contexts_{dataset_name}.jsonl', split='train')

#If I need a subset in the future, I may use :
#create_subset_from_context_dataset(dataset_name, subset_size=3000, random_seed=42)


#SFGRAM BIS
# now that there is a single .txt file
dataset_name = "sfgram"
jsonl_name = "sfgram"
if not file_already_exists("../source_datasets/", f"{jsonl_name}.jsonl"):
    dataset = load_dataset('text', data_files="../source_datasets/all_sf.txt", split='train')
    dataset.to_json(f"../source_datasets/{jsonl_name}.jsonl")
generate_contexts_from_jsonl_sentences(jsonl_name, dataset_name)
print("Generation of contexts finished")
