import sys

from utils.finetuning_utils import push_merged_model_to_hub
from utils.constants_utils import *


base_model_name = BASE_MODEL_NAME
finetuned_model_name = f"peft_model_18-22_{BASE_MODEL_NAME.split('/')[-1]}"

for revision in [f"stereotyped_{str(1e-5)}_{20}", f"stereotyped_related_{str(1e-5)}_{20}"]:
    push_merged_model_to_hub(base_model_name, finetuned_model_name, revision, verbose=True)
