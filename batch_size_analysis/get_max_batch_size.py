import sys

from utils.inference_utils import full_inference
from utils.constants_utils import *


model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
subset_size = SUBSET_SIZES[0]

# the random seed is used for the shuffling of the dataset before creating the inferred dataset ;
# it is not supposed to influence the inference

batch_sizes_to_test = [2**i for i in range(8)] + [24, 48]
batch_sizes_to_test.sort()

for batch_size in batch_sizes_to_test:
    res, loading_time, inference_time = full_inference(context_dataset_name, subset_size, batch_size,
                                                       model_name, random_seed)
    print(f"batch_size {batch_size} : inference time {inference_time} s ({inference_time/subset_size} s per context)")
