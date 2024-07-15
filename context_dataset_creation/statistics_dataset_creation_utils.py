import os
import matplotlib.pyplot as plt
import numpy as np


#MANAGING FILES

def get_list_of_filenames_from_path(path):
    """
    Returns the list of file names in a given path

    Parameters
    -----------
    path: str
        a string indicating the path of interest (it makes no difference if the string ends with '/' or not)

    Returns
    ___________
    dir_list: list[str]
        a list of the file names in the given path, with their extension
    """
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #We don't use just listdir because we want only files, not directories
    return filenames


def get_list_of_created_datasets():
    """
    Returns the list of created datasets.

    Returns
    ___________
    res: list[str]
        list of the file names of the created datasets without their extension (because it will necessarily be .json)
    """
    filenames_with_extension = get_list_of_filenames_from_path('../created_datasets')
    res = [filename[:-5] for filename in filenames_with_extension]
    return res


def get_list_of_created_statistics():
    """
    Returns the list of created statistics.

    Returns
    ___________
    res: list[str]
        list of the file names of the created statistics without their extension (because it will necessarily be .json)
    """
    filenames_with_extension = get_list_of_filenames_from_path('../saved_objects/context_dataset_creation/'
                                                               'created_statistics')
    res = [filename[:-5] for filename in filenames_with_extension]
    return res


def get_histogram_length(statistics_dictionary, filename_without_extension, max_len=51):
    name_output_file = "histogram" + filename_without_extension[10:]
    #print(name_output_file)

    length_contexts = np.array(statistics_dictionary["length_contexts"])
    nb_steps, nb_lengths = length_contexts.shape
    nb_lengths = min(nb_lengths, max_len)

    unfiltered = length_contexts[0]
    person_neutral = length_contexts[1]
    name_neutral = length_contexts[2]

    fig, ax = plt.subplots()
    p = ax.bar(np.arange(nb_lengths), unfiltered[:nb_lengths], label="unfiltered", bottom=np.zeros(nb_lengths))
    p = ax.bar(np.arange(nb_lengths), person_neutral[:nb_lengths], label="person-neutral", bottom=np.zeros(nb_lengths))
    p = ax.bar(np.arange(nb_lengths), name_neutral[:nb_lengths], label="person-neutral and name-neutral", bottom=np.zeros(nb_lengths))

    #ax.set_title("Number of contexts of a given length")
    ax.set_xlabel("Length of contexts")
    ax.set_ylabel("Number of contexts")
    ax.legend(loc="upper right")

    plt.savefig(f"../saved_objects/context_dataset_creation/created_visualizations/{name_output_file}.png", dpi=200)
    plt.close(fig)
