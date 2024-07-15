from datasets import load_dataset
from prettytable import PrettyTable

import sys

from utils.constants_utils import *
from utils.global_utils import prettytable_to_latex


#The code is very similar to adding_probs_to_subset.py

dama_model_names = [f"dama_model_{start_layer}-{start_layer + num_layers - 1}_Llama-2-7b-hf_stereotyped_related"
                    for start_layer in [18] for num_layers in [5, 6, 8, 10, 12, 14]]
peft_model_names = [f"PaulM2000/peft_model_18-22_Llama-2-7b-hf {train_data}_{str(lr)}_{20}"
                    for lr in (1e-5, 1e-6) for train_data in ("stereotyped_related", "stereotyped")]

model_names = [BASE_MODEL_NAME] + dama_model_names + peft_model_names
# cf. evaluate_bias_relevance

finetuning_dataset_name = f"finetuning_datasets/test_ft"
ft_dataset = load_dataset('json', data_files=f'../created_datasets/{finetuning_dataset_name}.jsonl',
                          split='train')

nb_sentences = 40
subset_size = 0

columns_names = ["Sentence"] + [f"{'_'.join(m.split('/')[-1].split())}" for m in model_names]
table = PrettyTable(columns_names)

for i in range(nb_sentences):
    row = []
    for m in model_names:
        jsonl_name = f"inferred_{finetuning_dataset_name.split('/')[-1]}_{subset_size}_{DATASET_RANDOM_SEED}_" \
                     f"{'_'.join(m.split('/')[-1].split())}"
        my_dataset = load_dataset('json', data_files=f'../created_datasets/inferred_datasets/'
                                                     f'{jsonl_name}.jsonl',
                                  split='train')
        if not row:
            row.append(my_dataset['text'][i])
        row.append((round(my_dataset['prob_he'][i], 3), round(my_dataset['prob_she'][i],3), round(my_dataset['bias_score'][i], 3)))

    table.add_row(row)

#The table is filled

#Now we can save it
chain = table.get_string()
# I can use the fields argument of get_string to easily keep only columns of interest
print(chain)

repertory = "../saved_objects/evaluation/he_she"
with open(f"{repertory}/he_she_table.txt", 'w') as f:
    f.write(chain)

prettytable_to_latex(f"{repertory}/he_she_table")
