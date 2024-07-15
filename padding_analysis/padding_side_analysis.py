from datasets import load_dataset

import sys

from utils.inference_utils import add_inference_to_subset, get_perpexity_from_names, get_top_k_predictions_from_strings, add_inference_to_subset2
from utils.global_utils import file_already_exists
from utils.constants_utils import *


#The code is very similar to adding_probs_to_subset.py

model_names = [BASE_MODEL_NAME,
               "dama_model_18-22_Llama-2-7b-hf_stereotyped_related",
               f"PaulM2000/peft_model_18-22_Llama-2-7b-hf stereotyped_related_{str(1e-5)}_{20}",
               #f"PaulM2000/peft_model_18-22_Llama-2-7b-hf stereotyped_{str(1e-6)}_{20}",
               #HUB_LIMI_MODEL_NAME + " " + LIMI_REVISION,
               #HUB_ALL_LINEAR_MODEL_NAME + " " + ALL_LINEAR_REVISION,
               ]

batch_size = BATCH_SIZE
finetuning_dataset_name = f"finetuning_datasets/test_ft"

for m in model_names:
    print(f"\n####################################\n{m}\n")
    # 0. Perplexity
    for pad_side in ["left", "right"]:
        ppl = get_perpexity_from_names(m, pad_side=pad_side)
        print(f"\nperplexity (padding side {pad_side}) : {ppl}\n")
