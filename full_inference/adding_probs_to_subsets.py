import sys

from utils.inference_utils import add_inference_to_subset
from utils.global_utils import file_already_exists
from utils.constants_utils import *


#print("CUDA available:", torch.cuda.is_available(), file=sys.stderr)
#print("CUDNN enabled:", torch.backends.cudnn.enabled, file=sys.stderr)
#print("GPUs found:", torch.cuda.device_count(), file=sys.stderr)
#What David uses on UFAL cluster is Cuda 11.8 and pytorch 2.1.
# I have Cuda 11.2 and Pytorch 2.1.1

model_names = MODEL_NAMES
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_sizes = SUBSET_SIZES
for subset_size in subset_sizes:
    for model_name in model_names:
        jsonl_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}" \
                     f"_{'_'.join(model_name.split('/')[-1].split())}"
        # The .join is necessary for finetuned models where space in the name ;
        # but it shouldn't change anything compared to old code for base model
        if not file_already_exists("../created_datasets/inferred_datasets/", f"{jsonl_name}.jsonl"):
            add_inference_to_subset(context_dataset_name, subset_size, batch_size,
                                    model_name=model_name,
                                    random_seed=random_seed, verbose=False)
        else:
            print(f"{jsonl_name}.jsonl already exists in ../created_datasets/inferred_datasets/")
