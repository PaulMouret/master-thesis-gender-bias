from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from blingfire import text_to_words

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.constants_utils import *


model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = 50000#SUBSET_SIZES[-1]

jsonl_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}" \
                 f"_{'_'.join(model_name.split('/')[-1].split())}"
#The .join is necessary for finetuned models where space in the name ;
# but it shouldn't change anything compared to old code for base model

#We don't need the original dataset inference was obtained from, except for its length :
my_dataset = load_dataset('json', data_files=f'../created_datasets/inferred_datasets/'
                                             f'{jsonl_name}.jsonl',
                          split='train')

print("\n###########\nLONG CONTEXTS")
long_dataset = my_dataset.filter(lambda x: len(text_to_words(x["text"]).split()) >= 15)
for example in long_dataset:
    print(f"{example['text']} ({example['relevance_score']}) ({example['original_text']})")

print("\n##########")

lengths = list(np.arange(1,30))
medians = []
bias_medians = []
bias_abs_medians = []
for i in range(1, 30):
    fixed_len_dataset = my_dataset.filter(lambda x: len(text_to_words(x["text"]).split()) == i)
    medians.append(100 * np.median(fixed_len_dataset['relevance_score']))
    bias_medians.append(np.median(fixed_len_dataset['bias_score']))
    bias_abs_medians.append(np.median(np.abs(fixed_len_dataset['bias_score'])))
    print(f"{i} : {len(fixed_len_dataset)} contexts : {np.median(fixed_len_dataset['relevance_score'])} median relevance {np.median(fixed_len_dataset['bias_score'])} median bias")

plt.plot(lengths, medians, marker='.')
plt.xlabel("Length of the context")
plt.ylabel("Median relevance score (in %)")
plt.savefig(f"../saved_objects/benchmark/median_relevance_of_length_plot.png", dpi=200)
plt.close()

plt.plot(lengths, bias_medians, marker='.')
plt.xlabel("Length of the context")
plt.ylabel("Median bias")
plt.savefig(f"../saved_objects/benchmark/median_bias_of_length_plot.png", dpi=200)
plt.close()

plt.plot(lengths, bias_abs_medians, marker='.')
plt.xlabel("Length of the context")
plt.ylabel("Median bias measure")
plt.savefig(f"../saved_objects/benchmark/median_bias_measure_of_length_plot.png", dpi=200)
plt.close()

nb = len(my_dataset)
nb_classical = len(my_dataset.filter(lambda x: not x["dialogue_context"]))
med_bias = np.median(my_dataset["bias_score"])

print("proportion of classical contexts : ", nb_classical/nb)
print(f"median bias : {med_bias}")

#irrelevant_classical = my_dataset.filter(lambda x: x["relevance_score"] < 0.001 and not x["dialogue_context"])
#irrelevant_dialogue = my_dataset.filter(lambda x: x["relevance_score"] < 0.001 and x["dialogue_context"])

#relevant_classical = my_dataset.filter(lambda x: x["relevance_score"] >= 0.04 and not x["dialogue_context"])
#relevant_dialogue = my_dataset.filter(lambda x: x["relevance_score"] >= 0.04 and x["dialogue_context"])

my_dataset = my_dataset.filter(lambda x: x["relevance_score"] >= RELEVANCE_THRESHOLD)
nb_relevant = len(my_dataset)
nb_classical_relevant = len(my_dataset.filter(lambda x: not x["dialogue_context"]))

print("proportion of classical contexts among relevant contexts : ", nb_classical_relevant/nb_relevant)


med_bias_relevant = np.median(my_dataset["bias_score"])
print(f"median bias among relevant contexts : {med_bias_relevant}")
