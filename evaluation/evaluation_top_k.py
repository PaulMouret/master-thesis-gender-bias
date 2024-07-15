from datasets import load_dataset
from prettytable import PrettyTable
import numpy as np

import sys

from utils.inference_utils import get_top_k_predictions_from_strings
from utils.global_utils import prettytable_to_latex
from utils.constants_utils import *


#The code is very similar to adding_probs_to_subset.py

model_names = [BASE_MODEL_NAME,
               "dama_model_18-22_Llama-2-7b-hf_stereotyped_related",
               f"PaulM2000/peft_model_18-22_Llama-2-7b-hf stereotyped_related_{str(1e-5)}_20"]

finetuning_dataset_name = f"finetuning_datasets/test_ft"
ft_dataset = load_dataset('json', data_files=f'../created_datasets/{finetuning_dataset_name}.jsonl',
                          split='train')

nb_sentences = 16 #be careful : if it's too big, should proceed to batching
k = 4
strings = ft_dataset["text"][:nb_sentences]

list_biggest_probabilities = []
list_most_probable_words = []

for m in model_names:
    biggest_probabilities, most_probable_words = get_top_k_predictions_from_strings(strings, m, k=k)
    list_biggest_probabilities.append(biggest_probabilities)
    list_most_probable_words.append(most_probable_words)

list_biggest_probabilities = np.array(list_biggest_probabilities)
list_most_probable_words = np.array(list_most_probable_words)
# They have shape (nb_models, nb_sentences, k)

#From there we can create a table
#For space reasons, it is better to have a row for each sentence

columns_names = ["Sentence"] + model_names
table = PrettyTable(columns_names)

for i in range(nb_sentences):
    res = [[(words[j], round(probs[j],3)) for j in range(k)] for (words, probs) in zip(list_most_probable_words[:,i], list_biggest_probabilities[:,i])]
    row = [strings[i]] + res
    #print(f"row : {row}")
    table.add_row(row)

#The table is filled

#Now we can save it
chain = table.get_string()
# I can use the fields argument of get_string to easily keep only columns of interest
print(chain)

repertory = "../saved_objects/evaluation/top_k"
with open(f"{repertory}/top_k_table.txt", 'w') as f:
    f.write(chain)

prettytable_to_latex(f"{repertory}/top_k_table")
