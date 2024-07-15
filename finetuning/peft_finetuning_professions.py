import numpy as np

import sys

sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
# This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.constants_utils import *
from utils.finetuning_utils import finetuning

# Note that the name of train and test set are constants, independent from any hyperparameter

learning_rates = [1e-5]
start_layers = [18]
num_layers = [8]#[5, 6, 8, 10, 12, 14]
include_gate_projs = [True, False]

for start_layer in start_layers:
    for num_layer in num_layers:
        for learning_rate in learning_rates:
            for include_gate_proj in include_gate_projs:
                str_gate_proj = '_with_gp' if include_gate_proj else ''
                end_layer = start_layer + num_layer - 1

                finetuned_model_name = f"peft_model_{start_layer}-{end_layer}{str_gate_proj}" \
                                       f"_{BASE_MODEL_NAME.split('/')[-1]}"
                revision = f"professions_{str(learning_rate)}_{20}"
                print(f"{finetuned_model_name} {revision}")

                # 3. Actual finetuning
                finetuning(base_model_name=BASE_MODEL_NAME,
                           finetuned_model_name=finetuned_model_name,
                           revision=revision,
                           train_dataset_name=f'train_dama_random_42', test_dataset_name=f"test_ft",
                           num_epochs=20, learning_rate=learning_rate,
                           batch_size=8, gradient_accumulation_steps=4,
                           num_layers=np.arange(start_layer, start_layer + num_layer),
                           include_up_proj=True, include_gate_proj=include_gate_proj,
                           lora_r=16, lora_alpha=16, lora_dropout=0.,
                           verbose=True)
