import numpy as np

import sys

from utils.constants_utils import *
from utils.finetuning_utils import finetuning


# 0. Name of the base dataset the test and train dataset come from
model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

# 1. Choice of the train dataset

for choice_stereotypical in CHOICES_STEREOTYPICAL:
    for strategy in FINETUNING_STRATEGIES:

        # 2. Creation of the names in consequence
        if choice_stereotypical:
            stereotype_str = "stereotyped"
        else:
            stereotype_str = "nonstereotyped"
        full_strategy = f'{stereotype_str}_{strategy}'

        # 3. Actual finetuning
        finetuning(base_model_name=BASE_MODEL_NAME,
                   finetuned_model_name=f"peft_model_all_linear_all_datasets_{BASE_MODEL_NAME.split('/')[-1]}",
                   revision=f"{full_strategy}_{str(AL_AD_LEARNING_RATE)}_{AL_AD_NUM_EPOCHS}",
                   train_dataset_name=f'train_{full_strategy}_{full_dataset_name}', test_dataset_name=f"test_{full_dataset_name}",
                   num_epochs=AL_AD_NUM_EPOCHS, learning_rate=AL_AD_LEARNING_RATE,
                   batch_size=8, gradient_accumulation_steps=4,
                   num_layers=np.arange(32), include_up_proj=True,
                   lora_r=16, lora_alpha=16, lora_dropout=0.,
                   verbose=True)
