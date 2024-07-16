from datasets import load_dataset, Dataset
import numpy as np
import pyarrow

import sys

from utils.dama_constants_utils import *
from utils.global_utils import load_obj_from_jsonfile


LLAMA_PRONOUNS = {"pos": "he",
                  "neg": "she",
                  "neut": "they"}

strategies = DAMA_STRATEGIES
random_seeds = DAMA_RANDOM_SEEDS

train_dama_json = load_obj_from_jsonfile("../source_datasets/", "train_dama")

for strategy in strategies:
    for random_seed in random_seeds:
        dataset_name = f"train_dama_{strategy}_{random_seed}"
        np.random.seed(random_seed)
        list_of_texts = []
        for example in train_dama_json:
            if strategy == "random":
                target = np.random.choice(list(LLAMA_PRONOUNS.values()))
            elif strategy == "neutral":
                target = LLAMA_PRONOUNS["neut"]
            elif strategy == "opposite":
                target = example["target_new"]["str"]

            list_of_texts.append(example["prompt"] + " " + target)

        finetuning_dataset = Dataset(pyarrow.Table.from_arrays([list_of_texts], names=["text"]))
        finetuning_dataset.to_json(f'../created_datasets/finetuning_dama_datasets/{dataset_name}.jsonl')
        finetuning_dataset.to_json(f'../created_datasets/finetuning_datasets/{dataset_name}.jsonl')
