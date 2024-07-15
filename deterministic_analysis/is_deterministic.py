from datasets import load_dataset
import torch
import numpy as np
import time

import sys

from utils.inference_utils import full_inference
from utils.constants_utils import *


model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[0]
# the random seed is used for the shuffling of the dataset before creating the inferred dataset ;
# it is not supposed to influence the inference

res_1, loading_time_1, inference_time_1 = full_inference(context_dataset_name, subset_size, batch_size,
                                                         model_name, random_seed)
res_2, loading_time_2, inference_time_2 = full_inference(context_dataset_name, subset_size, batch_size,
                                                         model_name, random_seed)

deterministic = (res_1 == res_2)
print(f"The model is deterministic : {deterministic}")

if not deterministic:
    res_1 = np.array(res_1).flatten()
    res_2 = np.array(res_2).flatten()

    matching_proportion = np.count_nonzero(res_1 == res_2) / len(res_2)
    print(f"matching_proportion : {matching_proportion}")
