from prettytable import PrettyTable
import os

from utils.global_utils import load_obj_from_jsonfile, prettytable_to_latex
from utils.constants_utils import *


# First, we compute beforehand the names of all models and their paths
path_results = '../saved_objects/evaluation/bias_relevance/'

dama_model_names = [f"dama_model_{start_layer}-{start_layer + num_layers - 1}_Llama-2-7b-hf_{train_dataset}"
                    for train_dataset in ["stereotyped_related", "stereotyped", "professions"]
                    for start_layer in [18] for num_layers in [5, 12]] #[5, 6, 8, 10, 12, 14]

peft_model_names = [f"PaulM2000/peft_model_{start_layer}-{start_layer + num_layer - 1}{str_gate_proj}" \
                    f"_{BASE_MODEL_NAME.split('/')[-1]} {train_data}_{str(learning_rate)}_{20}"
                    for start_layer in [18] for num_layer in [5, 12] for learning_rate in [1e-5]
                    for str_gate_proj in ['_with_gp', '']
                    for train_data in ("stereotyped_related", "stereotyped", "professions")] #[5, 6, 8, 10, 12, 14], ['_with_gp', '']

#model_names = [BASE_MODEL_NAME] + dama_model_names + peft_model_names
# cf. evaluate_bias_relevance

paths = [BASE_MODEL_NAME]
names = ["LLaMA 2 7B"]
start_layer = 18
learning_rate = 1e-5
for num_layers in [5, 12]: #[5, 6, 8, 10, 12, 14]
    for method in ["DAMA", "LoRA"]:
        for train_data in ["stereotyped_related", "stereotyped", "professions"]:
            if method == "DAMA":
                path = f"dama_model_{start_layer}-{start_layer + num_layers - 1}_Llama-2-7b-hf_{train_data}"
                name = f"{num_layers}L / DAMA / {train_data}"
                paths.append(path)
                names.append(name)
            elif method == "LoRA":
                for str_gate_proj in ['_with_gp']: #, ''
                    peft_model_name = f"PaulM2000/peft_model_{start_layer}-{start_layer + num_layers - 1}{str_gate_proj}" \
                    f"_{BASE_MODEL_NAME.split('/')[-1]} {train_data}_{str(learning_rate)}_{20}"
                    path = peft_model_name
                    name = f"{num_layers}L / LoRA / {train_data}" #{str_gate_proj}
                    paths.append(path)
                    names.append(name)

#The dataset we got our evalaution results from
finetuning_dataset_name = f"finetuning_datasets/test_ft"
subset_size = 0


# We make one table for each type of context (all, classical, dialogue)

for i_type, type_sentence in enumerate(["all", "dialogue", "classical"]):
    print(f"\n### {type_sentence}")
    columns_names = ["Model", "median prob he", "median prob she", "median relevance score",
                     "median bias score", "median bias measure"]
    key_names = ["Probability of 'he' (in %)_median", "Probability of 'she' (in %)_median",
                 "Relevance score_median", "Bias score_median", "Bias measure_median"]
    table = PrettyTable(columns_names)

    #Now we can fill-in the table
    for path, name in zip(paths, names):
        clean_model_name = f"{'_'.join(path.split('/')[-1].split())}"
        jsonl_name = f"inferred_{finetuning_dataset_name.split('/')[-1]}_{subset_size}_{DATASET_RANDOM_SEED}_" \
                     f"{clean_model_name}"
        filename = f"{jsonl_name}_bias_relevance"

        res = load_obj_from_jsonfile(path_results, filename)[i_type]
        row = [name] + [round(res[k], 3) if k in ["Bias score_median", "Bias measure_median"] else round(res[k], 1) for k in key_names]
        table.add_row(row)

    #The table is filled

    #Now we can save it
    chain = table.get_string()
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/evaluation/bias_relevance"
    with open(f"{repertory}/bias_relevance_{type_sentence}_table.txt", 'w') as f:
        f.write(chain)

    prettytable_to_latex(f"{repertory}/bias_relevance_{type_sentence}_table", first_column_bold=True)
