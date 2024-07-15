from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr
from collections import Counter
from blingfire import text_to_words
from prettytable import PrettyTable

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile, print_full_descriptive_statistics, prettytable_to_latex
from utils.constants_utils import *


# I - Creation of useful objects

# 0. Load dataset

model_name = BASE_MODEL_NAME
context_dataset_name = CONTEXT_DATASET_NAME
random_seed = DATASET_RANDOM_SEED
batch_size = BATCH_SIZE
subset_size = SUBSET_SIZES[-1]
full_dataset_name = f"inferred_{context_dataset_name}_{subset_size}_{random_seed}_{model_name.split('/')[-1]}"

corrupted_dataset_name = f'{full_dataset_name}_final_noising_experiment'

corrupted_dataset = load_dataset('json',
                                 data_files=f'../created_datasets/noising_experiment_datasets/'
                                            f'{corrupted_dataset_name}.jsonl',
                                 split='train')

# 1. We load the dictionaries
dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                        f"dict_bias_diff_{subset_size}")
dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                     f"dict_counts_{subset_size}")
dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                   f"dict_stereotypical_scores_{subset_size}")

#And similarly for type-specific-dictionaries
dialogue_dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                 f"dialogue_dict_bias_diff_{subset_size}")
dialogue_dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                              f"dialogue_dict_counts_{subset_size}")
dialogue_dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                            f"dialogue_dict_stereotypical_scores_{subset_size}")

classical_dict_bias_diff = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                  f"classical_dict_bias_diff_{subset_size}")
classical_dict_counts = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                               f"classical_dict_counts_{subset_size}")
classical_dict_stereotypical_scores = load_obj_from_jsonfile("../saved_objects/noising_experiment/stereotypical_scores/",
                                                             f"classical_dict_stereotypical_scores_{subset_size}")

# II - Analaysis

NB_MOST_COMMON = 120
counter_counts = Counter(dict_counts)
nb_words_with_duplicate = counter_counts.total()
print(f"\nWe analyze {corrupted_dataset_name}.")
print(f"There are {nb_words_with_duplicate} words (with duplicate) in total.")
nb_sentences = len(corrupted_dataset)
print(f"There are {nb_sentences} sentences in total.")

# 1. Frequency of words and choice of a frequency threshold
print("\n\n###########\n1. Frequency of words and choice of a frequency threshold")
frequency_threshold = OCCURRENCE_THRESHOLD

#Frequent contexts
print("\nFrequent contexts (ie. whose all words meet the threshold)")
def min_counts_in_context(context):
    return min([dict_counts[word] for word in text_to_words(context["text"]).lower().split()])
frequent_dataset = corrupted_dataset.filter(lambda x: min_counts_in_context(x) >= frequency_threshold)
print(f"There are {len(frequent_dataset)} ({len(frequent_dataset)/len(corrupted_dataset)}) frequent contexts.")

print(f"\n#######\nDistribution of stereotypical scores in person-hyponyms")
clean_gender_of_persons = load_obj_from_jsonfile('../created_datasets/utils_datasets/', "clean_gender_of_persons")
neutral_persons = [person['word'] for person in clean_gender_of_persons if person['gender'] == 'n']
print(f"There are {len(neutral_persons)} neutral person-hyponyms (in language).")
neutral_stereotypical_scores = [dict_stereotypical_scores[w] for w in neutral_persons
                                if w in dict_counts.keys() and dict_counts[w] >= frequency_threshold]
print(f"There are {len(neutral_stereotypical_scores)} neutral person-hyponyms that meet the occurrence threshold.")
neutral_stereotypical_scores = np.array(neutral_stereotypical_scores)
#Distribution
print_full_descriptive_statistics(neutral_stereotypical_scores, "neutral_stereotypical_scores", more_quantiles=True)
#Histogram
plt.hist(neutral_stereotypical_scores, bins=100, density=True) #range
plt.xlabel("Neutral person-hyponyms stereotypical scores")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/stereotypical_scores/histogram_neutral_stereotypical_scores.png",
            dpi=200)
plt.close()

# 2. Stereotypical scores and choice of (male and female) stereotype thresholds
print("\n\n###########\n2. Stereotypical scores and choice of (male and female) stereotype thresholds")

#Words with most stereotypes (ie. largest and lowest stereotypical scores)
print("\nWords with most stereotypes (without frequency threshold) :")
counter_male_stereotype = Counter(dict_stereotypical_scores)
counter_female_stereotype = Counter({word: -(dict_stereotypical_scores[word])
                                     for word in dict_stereotypical_scores.keys()})
print("Male stereotypes :\n", counter_male_stereotype.most_common(NB_MOST_COMMON))
print("Female stereotypes (in abs. value)\n: ", counter_female_stereotype.most_common(NB_MOST_COMMON))
#the dict does not really contain counts, but the method still returns the keys with highest values

#Words with most stereotypes for the chosen frequency threshold, and their score
print(f"Words with most stereotypes (for the chosen threshold {frequency_threshold})")
counter_male_stereotype_with_threshold = Counter({word: dict_stereotypical_scores[word]
                                                  for word in dict_stereotypical_scores.keys()
                                                  if dict_counts[word] >= frequency_threshold})
counter_female_stereotype_with_threshold = Counter({word: -(dict_stereotypical_scores[word])
                                                    for word in dict_stereotypical_scores.keys()
                                                    if dict_counts[word] >= frequency_threshold})
print("Male stereotypes :\n", counter_male_stereotype_with_threshold.most_common(NB_MOST_COMMON))
print("Female stereotypes (in abs. value)\n: ",
      counter_female_stereotype_with_threshold.most_common(NB_MOST_COMMON))

for counter, gender in zip([counter_male_stereotype_with_threshold, counter_female_stereotype_with_threshold],
                           ["male", "female"]):
    table = PrettyTable(["word", "stereotypical score"])
    for word, score in counter.most_common(42): #number of lines in table
        if gender == "female":
            score = (- score)
        table.add_row([word, round(score, 3)])

    # The table is filled
    # Now we can save it
    chain = table.get_string()
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/noising_experiment/stereotypical_scores"
    with open(f"{repertory}/table_most_{gender}_stereotyped.txt", 'w') as f:
        f.write(chain)
    prettytable_to_latex(f"{repertory}/table_most_{gender}_stereotyped")

#The full list in text form
for counter, gender in zip([counter_male_stereotype_with_threshold, counter_female_stereotype_with_threshold],
                           ["male", "female"]):
    results_directory = "../saved_objects/noising_experiment/stereotypical_scores/"
    tabout = open(f"{results_directory}{gender}_strong_stereotypes_latex.txt", 'w')
    for word, score in counter.most_common(200): #N should be large enough to include all strong stereotypes
        if gender == "female":
            score = (- score)
        if score <= FEMALE_THRESHOLD or score >= MALE_THRESHOLD:
            word = word.replace("_", "\_")
            text_chain = f"\\context{{{word}}} (${round(score, 3)}$), "
            tabout.write(text_chain)
# SHOULD BE UPDATED IF I WANT TO PRINT THE FULL LIST OF STRONG STEREOTYPES

#Distribution of stereotypical scores
print("\nDistribution of stereotypical scores")
stereotypical_scores = np.array([score for score in dict_stereotypical_scores.values()])
print_full_descriptive_statistics(stereotypical_scores, "stereotypical_scores", more_quantiles=True)
#Histogram
plt.hist(stereotypical_scores, bins=100, range=(-2, 2), density=True)
plt.xlabel("Stereotypical scores")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/stereotypical_scores/histogram_stereotypical_scores.png", dpi=200)
plt.close()

#Distribution of frequent stereotypical scores
print("\nDistribution of frequent stereotypical scores")
frequent_stereotypical_scores = np.array([dict_stereotypical_scores[word] for word in dict_stereotypical_scores.keys()
                                          if dict_counts[word] >= frequency_threshold])
print_full_descriptive_statistics(frequent_stereotypical_scores, "frequent_stereotypical_scores", more_quantiles=True)
print(f"Proportion of frequent stereotypical words : {len(frequent_stereotypical_scores)/len(stereotypical_scores)}")
#Histogram
plt.hist(frequent_stereotypical_scores, bins=100, range=(-2, 2), density=True)
plt.xlabel("Frequent stereotypical scores")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/stereotypical_scores/histogram_frequent_stereotypical_scores.png",
            dpi=200)
plt.close()

#Final choice
female_threshold = -0.4367205466568182
male_threshold = 0.39177736709879446

#Testing the chosen stereotypes thresholds
print(f"\nTesting the chosen stereotypes thresholds (female {female_threshold}, male {male_threshold}) :")
frequent_male_stereotypes = [word for word in dict_stereotypical_scores.keys()
                    if dict_stereotypical_scores[word] > male_threshold
                    and dict_counts[word] > frequency_threshold]
frequent_female_stereotypes = [word for word in dict_stereotypical_scores.keys()
                    if dict_stereotypical_scores[word] < female_threshold
                    and dict_counts[word] > frequency_threshold]
print(f"\n(frequent) male stereotypes ({len(frequent_male_stereotypes)}) :\n{frequent_male_stereotypes}")
print(f"\n(frequent) female stereotypes ({len(frequent_female_stereotypes)}) :\n{frequent_female_stereotypes}")

#Stereotypical contexts
print("\nStereotypical contexts")
def stereotype_in_context(context):
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()
              if dict_counts[word] >= frequency_threshold]
    if scores:
        female_score, male_score = min(scores), max(scores)
        return (female_score < female_threshold) or (male_score > male_threshold)
    else:
        return False

def all_female_stereotype_in_context(context):
    scores = [dict_stereotypical_scores[word] for word in text_to_words(context["text"]).lower().split()
              if dict_counts[word] >= frequency_threshold]
    if scores:
        return np.all(np.array(scores) < (-0.1)) #we want them "really" female stereotyped
    else:
        return False

stereotypical_dataset = corrupted_dataset.filter(lambda x: stereotype_in_context(x))
print(f"There are {len(stereotypical_dataset)} "
      f"({len(stereotypical_dataset)/len(corrupted_dataset)}) stereotypical contexts.")

stereotypical_and_frequent_dataset = frequent_dataset.filter(lambda x: stereotype_in_context(x))
print(f"There are {len(stereotypical_and_frequent_dataset)} "
      f"({len(stereotypical_and_frequent_dataset)/len(corrupted_dataset)}) stereotypical and frequent contexts.")

print("\nSome stereotypical contexts :")
for i in range(20):
    context = stereotypical_dataset[i]
    print(f"{context['text']} ({context['prob_he']}, {context['prob_she']})")

print("\nSome stereotypical contexts with frequent words :")
for i in range(20):
    context = stereotypical_and_frequent_dataset[i]
    print(f"{context['text']} ({context['prob_he']}, {context['prob_she']})")

# 3. Concrete examples and debugging
print("\n##########\n3. Concrete examples and debugging")

print("Examples of contexts with female stereotyped words and male bias")
paradox_dataset = frequent_dataset.filter(lambda x: all_female_stereotype_in_context(x) and x["bias_score"] > 0.2)
for i in range(20):
    print(f"{paradox_dataset['text'][i]} : bias {paradox_dataset['bias_score'][i]}")

#Examples of contexts
print("\nExamples of contexts")
specific_words = ["kid", "kids", "lipstick"]
for specific_word in specific_words:
    print(f"\n{specific_word} ({dict_counts[specific_word]} occurrences) :")
    word_dataset = corrupted_dataset.filter(lambda context: specific_word in text_to_words(context["text"]).lower().split())
    for context in word_dataset:
        for i, w in enumerate(text_to_words(context["text"]).lower().split()):
            if w == specific_word:
                print(f"{context['text']} ({i}) ({context['original_text']}) : "
                      f"prob_he {context['prob_he']} ; prob_she {context['prob_she']} ; "
                      f"relevance {context['relevance_score']} ; bias {context['bias_score']}")

#Example of stereotypical scores
print("\nExample of stereotypical scores")
for word in specific_words:
    print(f"{word} : {dict_stereotypical_scores[word]}")


# III - Comparison of the different types of stereotypical scores
print("\n#####################\nComparison of the different types of stereotypical scores\n")
for list_stereotype, name in []: #zip([frequent_male_stereotypes, frequent_female_stereotypes], ['male', 'female']):
    print(f"\n{name} stereotypes :")
    for word in list_stereotype:
        dialogue_word = word in dialogue_dict_counts.keys() and dialogue_dict_counts[word] >= frequency_threshold/2
        classical_word = word in classical_dict_counts.keys() and classical_dict_counts[word] >= frequency_threshold/2
        dialogue_score = dialogue_dict_stereotypical_scores[word] if dialogue_word else None
        classical_score = classical_dict_stereotypical_scores[word] if classical_word else None
        score = dict_stereotypical_scores[word]
        if dialogue_score and classical_score:
            dialogue_diff = dialogue_score - classical_score
            dialogue_str = "(more M when dialogue)" if dialogue_diff > 0 else "(more F when dialogue)"
            interesting = abs(classical_score - dialogue_score) > 0.3
        else:
            dialogue_str = ''
            interesting = False

        print(f"{word} : {score} / dialogue {dialogue_score} / classical {classical_score} "
              f"{'## INTERESTING ##' if interesting else ''} {dialogue_str}")

#Observations

#Over 2, probably a factually gendered word
#5e-05 seems to be a good threshold, but there remain a few first names ; 7e-05 is too restrictive
#kids is female stereotyped and kid is male stereotyped
# doc is male stereotyped but dr. is female stereotyped : the reason may be that dr. is followed by a second name,
#that may be stereotyped itself
#ari is a boy first name but was heavily female stereotyped ; probably because of more common first names like ariane
#a clue that it is a good measure : 'fucking' and 'f**ing' have very similar scores
# usually because factual info in sentence
#also factually gendered word that are hidden in other words (typically linked by a -)
#everything related to speech is female stereotyped : punctuation, and 'gon na', 'ai nt' which are verbal locutions
#insults/familiar words are male stereotyped : fucking, f**king, shit, damn
#frequency threshold has 2 uses : avoid first names ; guarantee a reliable score
#pregnant is a female stereotype ; but as it is linked to biological sex rather than gender,
# cannot be seen as a factually gendered word
#it looks like men tend to be associated with person-hyponyms (typically professions),
# while women are associated with characteristics

#Note that, after having removed most of first names, there remain some very stereotyped first names
#However, despite these stereotypes, they are not factually gendered
#note that factually gendered first names can be hidden in their possessive forms, where their tokenizing is so that
#the s is glued to the first name and makes it unrecognizable
#The word chapter is quite high in the female stereotypes, but actually it does not really play any role in sentence :
#it appears at the number of the chapter that precedes the first sentence of the chapter

#some weird input paired with tokenization errors give weird words, like ".i" and "_it"