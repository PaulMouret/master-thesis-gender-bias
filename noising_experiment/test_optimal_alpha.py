from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.constants_utils import *


relevance_threshold = RELEVANCE_THRESHOLD
factors = list(np.arange(1, 5.1, 0.1))
factors = [round(x, 2) for x in factors]
#copied from find_optimal_alpha.py (first experiment)
#We remove 100. for the purpose of plotting

model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
subset_size = 50000 #copied from find_optimal_alpha.py
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"
max_contexts = 1000 #copied from find_optimal_alpha.py


#We print the proportion of relevant noised contexts for alpha = 100
#The processing is slightly different because there may be NaN
full_corrupted_factor_dataset_name = f'{full_dataset_name}_{max_contexts}_corrupted_factor_100.0'
full_corrupted_factor_dataset = load_dataset('json',
                                        data_files=f'../created_datasets/noising_experiment_datasets/'
                                                   f'{full_corrupted_factor_dataset_name}.jsonl',
                                        split='train')
full_corrupted_relevances = np.array(
    [cr for context_crs in full_corrupted_factor_dataset["corrupted_relevances"] for cr in context_crs
     if cr is not None and not np.isnan(cr)])
print(f"corrupted_relevances :\n{full_corrupted_relevances}")
relevant_proportion = np.count_nonzero(full_corrupted_relevances >= relevance_threshold) / len(full_corrupted_relevances)
print(f"nb of relevant contexts at the origin : {len(full_corrupted_factor_dataset)}")
print(f"nb of corrupted relevances : {len(full_corrupted_relevances)}")
print(f"factor 100.0 : relevant proportion {relevant_proportion}")


relevant_proportions = []
for factor in factors:
    corrupted_factor_dataset_name = f'{full_dataset_name}_{max_contexts}_corrupted_factor_{factor}'

    corrupted_factor_dataset = load_dataset('json',
                                            data_files=f'../created_datasets/noising_experiment_datasets/'
                                                       f'{corrupted_factor_dataset_name}.jsonl',
                                            split='train')
    #We flatten corrupted relevances
    corrupted_relevances = np.array(
        [cr for context_crs in corrupted_factor_dataset["corrupted_relevances"] for cr in context_crs])
    relevant_proportion = np.count_nonzero(corrupted_relevances >= relevance_threshold) / len(corrupted_relevances)
    relevant_proportions.append(relevant_proportion)
    print("\n")
    print(f"nb of relevant contexts at the origin : {len(corrupted_factor_dataset)}")
    print(f"nb of corrupted relevances : {len(corrupted_relevances)}")
    print(f"factor {factor} : relevant proportion {relevant_proportion}")

#We plot it
plt.plot(factors, relevant_proportions, marker='.')
plt.xlabel("Gamma")
plt.ylabel("Proportion of relevant noised contexts")
plt.savefig(f"../saved_objects/noising_experiment/alpha_experiments/prop_relevant_noised_contexts.png", dpi=200)
plt.close()
