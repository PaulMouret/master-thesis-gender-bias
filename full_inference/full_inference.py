import torch

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.inference_utils import full_inference
from utils.global_utils import file_already_exists, store_obj_in_jsonfile
from utils.constants_utils import *


model_names = MODEL_NAMES
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_sizes = SUBSET_SIZES

for subset_size in subset_sizes:
    for model_name in model_names:
        name_inference = f"{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}" \
                         f"_full_inference"
        if not torch.cuda.is_available():
            name_inference += "_on_cpu"

        print(f"Doing {name_inference}")

        if not file_already_exists("../saved_objects/full_inference/", f"{name_inference}.json"):
            res, loading_time, inference_time = full_inference(context_dataset_name, subset_size,
                                                               batch_size, model_name, random_seed)
            my_dict = {'full_inference': res, 'inference_time': inference_time,
                       'loading_time': loading_time,
                       'dataset_name': context_dataset_name, 'subset_size': subset_size,
                       'batch_size': batch_size, 'model_name': model_name, 'random_seed': random_seed}

            store_obj_in_jsonfile(my_dict, "../saved_objects/full_inference/", name_inference)
        else:
            print(f"{name_inference}.json already exists in ../saved_objects/full_inference/")
