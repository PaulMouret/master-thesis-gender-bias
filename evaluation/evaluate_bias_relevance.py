from datasets import load_dataset

import sys

from utils.inference_utils import add_inference_to_subset, get_perpexity_from_names, get_top_k_predictions_from_strings
from utils.global_utils import file_already_exists, store_obj_in_jsonfile
from full_inference.benchmark_utils import full_inference_analysis
from utils.constants_utils import *


print_results = True

dama_model_names = [f"dama_model_{start_layer}-{start_layer + num_layers - 1}_Llama-2-7b-hf_{train_dataset}"
                    for train_dataset in ["stereotyped_related", "stereotyped", "professions"]
                    for start_layer in [18] for num_layers in [5, 6, 8, 10, 12, 14]]

peft_model_names = [f"PaulM2000/peft_model_{start_layer}-{start_layer + num_layer - 1}{str_gate_proj}" \
                    f"_{BASE_MODEL_NAME.split('/')[-1]} {train_data}_{str(learning_rate)}_{20}"
                    for start_layer in [18] for num_layer in [5, 6, 8, 10, 12, 14] for learning_rate in [1e-5]
                    for str_gate_proj in ('_with_gp', '')
                    for train_data in ("stereotyped_related", "stereotyped", "professions")]

model_names = dama_model_names + peft_model_names + [BASE_MODEL_NAME]

batch_size = BATCH_SIZE
finetuning_dataset_name = f"finetuning_datasets/test_ft"

for m in model_names:
    print(f"\n####################################\n{m}\n")

    # 1. Inference
    subset_size = 0
    jsonl_name = f"inferred_{finetuning_dataset_name.split('/')[-1]}_{subset_size}_{DATASET_RANDOM_SEED}_" \
                 f"{'_'.join(m.split('/')[-1].split())}"
    #print("jsonl_name : ", jsonl_name)
    # The .join is necessary for finetuned models where space in the name ;
    # but it shouldn't change anything compared to old code for base model
    if not file_already_exists("../created_datasets/inferred_datasets/", f"{jsonl_name}.jsonl"):
        add_inference_to_subset(finetuning_dataset_name, subset_size, batch_size,
                                model_name=m,
                                random_seed=DATASET_RANDOM_SEED, verbose=True)
    else:
        print(f"{jsonl_name}.jsonl already exists in ../created_datasets/inferred_datasets/")

    if print_results:

        #Now that the file exists, we do the benchmark (without saving pictures because it would be useless and costly)
        my_dataset = load_dataset('json', data_files=f'../created_datasets/inferred_datasets/'
                                                     f'{jsonl_name}.jsonl',
                                  split='train')

        list_dict = []

        for subdataset_name in [jsonl_name,
                                "dialogue_" + jsonl_name, "classical_" + jsonl_name, # can comment it for less detailed res
                                ]:
            if subdataset_name == "dialogue_" + jsonl_name:
                subdataset = my_dataset.filter(lambda x: x["dialogue_context"])
            elif subdataset_name == "classical_" + jsonl_name:
                subdataset = my_dataset.filter(lambda x: not x["dialogue_context"])
            elif subdataset_name == jsonl_name:
                subdataset = my_dataset
            else:
                raise Exception(f"Unknown subdataset name : {subdataset_name}")

            lengths = [len(example['text'].split()) for example in subdataset]

            print(f"\n##\n{subdataset_name}\n")
            dict_res = full_inference_analysis(subdataset['prob_he'], subdataset['prob_she'], subdataset['bias_score'],
                                               subdataset['relevance_score'], lengths, subdataset_name,
                                               save_pictures=False, return_dict=True)
            dict_res["inference_name"] = subdataset_name
            list_dict.append(dict_res)

        filename = f"{jsonl_name}_bias_relevance"
        store_obj_in_jsonfile(list_dict, '../saved_objects/evaluation/bias_relevance/', filename)
