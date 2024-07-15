from datasets import load_dataset

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import file_already_exists
from dataset_creation_utils import generate_contexts_from_jsonl_sentences, filter_dataset

#We create a context dataset

#We will use load_dataset(), so we will obtain Dataset instances
#In particular dataset['text'] is a list of strings
#These strings correspond to different things depending on datasets, so they will be described for each of them

#BOOKCORPUS
#a string is a sentence, lowercase, where different words, including punctuation are separated by a space
#Note that dialogues of the form : ''blabla'' she said are considered as two sentences
#It contains 74 004 228 strings
#We will use jsonl format from here (for some reason, with txt file, for similar code for statistics
#it raises Memory error The jsonl file is larger but not dramatically (5G instead of 4G)

dataset_name = "bookcorpus"
jsonl_name = "bookcorpus"
if not file_already_exists("../source_datasets/", f"{jsonl_name}.jsonl"):
    dataset = load_dataset('text', data_files="../source_datasets/bookcorpus.txt", split='train')
    dataset.to_json(f"../source_datasets/{jsonl_name}.jsonl")
generate_contexts_from_jsonl_sentences(jsonl_name, dataset_name)
print("Generation of contexts finished")

#in the future, for instance to put them in pipelines, the created dataset can be opened as a dataset object :
#my_dataset = load_dataset('json', data_files=f'created_datasets/contexts_{dataset_name}.jsonl', split='train')

#If I need a subset in the future, I may use :
#create_subset_from_context_dataset(dataset_name, subset_size=3000, random_seed=42)
