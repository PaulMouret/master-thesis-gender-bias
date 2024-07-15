import json
import os
import numpy as np
from scipy.stats import percentileofscore


REGEX_METACHARACTERS = {'.': '\.', '^': '\^', '$': '\$', '*': '\*', '+': '\+', '?': '\?', '{': '\{',
                        '}': '\}', '[': '\[', ']': '\]', '|': '\|', '(': '\(', ')': '\)'}
#only the slash is missing
#When I use re.finditer where the substring is a metacharacter, this may cause error, so I have to replace
#the metacharacters by their "escaped version"


#OPERATIONS ON STRINGS
def plural_form(word):
    if word[-1] in "szx" or word[-2:] in ["sh", "ch"]:
        plural = word + "es"
    elif word[-1] == "y" and word[-3:] != "boy" and word[-3:] != "guy":
        plural = word[:-1] + "ies"
    elif word[-3:] == "man":
        plural = word[:-3] + "men"
    else:
        plural = word + "s"
    return plural


def possessive_form(firstname):
    return firstname + "s"


def real_string_from_tokenized(sentence):
    # 1. We uppercase the first letter
    sentence = sentence[0].upper() + sentence[1:]
    # 2. We replace i by I
    while "i" in sentence.split():
        l = sentence.split()
        index_i = l.index("i")
        sentence = " ".join(l[:index_i]) + " I " + " ".join(l[index_i + 1:])
    # 3. We glue the commas and points to the preceding word
    while "," in sentence.split():
        l = sentence.split()
        index_comma = l.index(",")
        sentence = " ".join(l[:index_comma]) + ", " + " ".join(l[index_comma + 1:])
    while "." in sentence.split():
        l = sentence.split()
        index_comma = l.index(".")
        sentence = " ".join(l[:index_comma]) + ". " + " ".join(l[index_comma + 1:])

    return sentence


#FILE MANAGEMENT

def file_already_exists(path, filename):
    """
    Indicates if the specified file already exists in the specified path

    Parameters
    -----------
    path: str
        a string ending with '/', indicating the path of interest
    filename: str
        the name of the file of interest, including its extension

    Returns
    ___________
    already_exists: bool
        A boolean indicating if the specified file already exists in the specified path
    """
    dir_list = os.listdir(path)  # dir_list contains the list of all files of the path
    already_exists = False
    for name in dir_list:
        if name == filename:
            already_exists = True
    return already_exists


def store_obj_in_jsonfile(obj, path, filename_without_extension):
    """
    Stores a given object in a given path wih a given file name.

    Parameters
    -----------
    obj:
        the python object to store
    path: str
        a string ending with '/', indicating the path of interest
    filename_without_extension: str
        the name of the file of interest, without its extension (because it will necessarily be .json)
    """
    json_filename = filename_without_extension + ".json"
    #if file_already_exists(path, json_filename):
        #print(f"File {json_filename} already exists in directory {path}")
    final_filename = path + json_filename
    with open(final_filename, 'w') as f:
        json.dump(obj, f)


def load_obj_from_jsonfile(path, filename_without_extension):
    """
    Loads an object from a jsonfile with a given file name, at a given path

    Parameters
    -----------
    path: str
        a string ending with '/', indicating the path of interest
    filename_without_extension: str
        the name of the file of interest, without its extension (because it will necessarily be .json)

    Returns
    ___________
    obj:
        the python object that is loaded
        (an exception is returned if the given file name does not exist in the given directory)
    """
    json_filename = filename_without_extension + ".json"
    final_filename = path + json_filename
    with open(final_filename, 'r') as f:
        obj = json.load(f)
    return obj


def prettytable_to_latex(prettytable_directory, first_column_bold=False):
    tabin = open(f"{prettytable_directory}.txt", 'r')
    tabout = open(f"{prettytable_directory}_latex.txt", 'w')

    first_content_line = tabin.readlines()[1]  # readlines gets at the end of the file, so need to reopen it
    nb_columns = first_content_line.count("|") - 1
    desc_column = "|c" * nb_columns + "|"

    tabout.write(f"\\begin{{table}}[!h]\n\\centering\n\\begin{{tabular}}{{{desc_column}}}\n\hline\n")


    tabin = open(f"{prettytable_directory}.txt", 'r')
    for line in tabin:
        if line[0] != '+':
            # replace _ with \_
            line = line.replace("_", "\_")
            # remove fisrt and last | and add \\ at the end
            line = line[1:-2] + " \\\\ \\hline" + "\n"
            # replace remaining | with &
            if first_column_bold:
                columns = line.split("|")
                columns[0] = f"\\textbf{{{columns[0].strip()}}}"
                line = "|".join(columns)
            line = line.replace("|", " & ")
            tabout.write(line)

    tabout.write("\\end{tabular}\n\\caption{\\label{my-label}My title}\n\\end{table}")

    tabin.close()
    tabout.close()

    print(f"{prettytable_directory}_latex.txt created.")


# STATISTICAL ANALYSIS

def print_descriptive_statistics(vector, vector_name, cdf_values=[], return_dict=False, more_quantiles=False):
    print(f"\n{vector_name}")
    print(f"number of values : {len(vector)}")
    quantiles = 10 * np.arange(0, 11)
    if more_quantiles:
        quantiles = 10 * np.arange(0, 10.5, 0.5)
    percentiles = np.percentile(vector, quantiles)
    print(f"mean : {np.mean(vector)} ; median : {np.median(vector)} ; min : {np.min(vector)} ; max : {np.max(vector)}"
          f" ; std : {np.std(vector)}")
    print(f"percentiles : {[(q, p) for q, p in zip(quantiles, percentiles)]}")
    for value in cdf_values:
        print(f"{value} is the {percentileofscore(vector, value)}% quantile")

    if return_dict:
        res = {f"{vector_name}_mean": np.mean(vector), f"{vector_name}_median": np.median(vector),
               f"{vector_name}_min": np.min(vector), f"{vector_name}_max": np.max(vector),
               f"{vector_name}_std": np.std(vector)}
        return res


def print_full_descriptive_statistics(vector, vector_name, cdf_values=[], abs_cdf_values=[], return_dict=False,
                                      more_quantiles=False):
    res = print_descriptive_statistics(vector, vector_name, cdf_values, return_dict=return_dict,
                                       more_quantiles=more_quantiles)
    # We treat the case there may be negative values ; and analyse the distribution of the absolute value :
    if np.any(vector < 0):
        abs_vector = np.abs(vector)
        abs_name = f"abs_{vector_name}"
        res_abs = print_descriptive_statistics(abs_vector, abs_name, abs_cdf_values, return_dict=return_dict,
                                               more_quantiles=more_quantiles)
        if return_dict:
            res.update(res_abs)
    if return_dict:
        return res


# BIAS

def bias_score(probs_he, probs_she):
    return np.log2(np.array(probs_he) / np.array(probs_she))


def bias_measure(probs_he, probs_she):
    return np.abs(np.log(np.array(probs_he) / np.array(probs_she)))


def relevance_score(probs_he, probs_she):
    return np.add(probs_he, probs_she)


def bias_score_from_couple(res):
    res = np.array(res)
    res_he = res[:, 0].squeeze()
    res_she = res[:, 1].squeeze()
    return bias_score(res_he, res_she)


def relevance_score_from_couple(res):
    res = np.array(res)
    res_he = res[:, 0].squeeze()
    res_she = res[:, 1].squeeze()
    return relevance_score(res_he, res_she)
