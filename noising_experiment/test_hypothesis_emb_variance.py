from datasets import load_dataset
import time
import torch
import numpy as np

import sys

from utils.constants_utils import *
from utils.global_utils import store_obj_in_jsonfile
from utils.noising_utils import ModelAndTokenizer, collect_embedding_std_for_sigma
from utils.inference_utils import load_tokenizer


tokenizer = load_tokenizer(BASE_MODEL_NAME)
vocab_size = tokenizer.vocab_size

start = time.time()

encoded_words_set = list(np.arange(vocab_size))

torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
mt = ModelAndTokenizer(BASE_MODEL_NAME, torch_dtype=torch_dtype)
print("Model loaded")

#The final value we keep for std
std_data, final_std = collect_embedding_std_for_sigma(mt, encoded_words_set)
print(f"final std : {final_std}")

end = time.time()
print(f"{end - start} s for {vocab_size} tokens")

store_obj_in_jsonfile(std_data,
                      '../saved_objects/noising_experiment/sigma_experiments/', "std_data")
