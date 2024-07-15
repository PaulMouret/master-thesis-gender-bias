import numpy as np

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile
from statistics_dataset_creation_utils import get_list_of_created_statistics, get_histogram_length


context_statistics_files = get_list_of_created_statistics()

#To print the mean length
for filename_without_extension in context_statistics_files:
    statistics_dictionary = load_obj_from_jsonfile("../saved_objects/context_dataset_creation/created_statistics/",
                                                   filename_without_extension)

    length_contexts = np.array(statistics_dictionary["length_contexts"])
    print(f"\n{filename_without_extension}")
    for i, stage in enumerate(length_contexts):
        mean_length = np.sum([k * stage[k] for k in range(len(stage))]) / np.sum(stage)
        print(f"stage {i} : mean length {mean_length}")

#To create the visualizations :
for filename_without_extension in context_statistics_files:
    statistics_dictionary = load_obj_from_jsonfile("../saved_objects/context_dataset_creation/created_statistics/",
                                                   filename_without_extension)
    get_histogram_length(statistics_dictionary, filename_without_extension)
