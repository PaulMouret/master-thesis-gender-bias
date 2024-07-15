import numpy as np

import sys

from utils.global_utils import load_obj_from_jsonfile, store_obj_in_jsonfile, plural_form


#Creates the datasets and statistics related to gender_of_persons

#IMPORTATION AND DATASET CREATION

gender_of_persons_original = load_obj_from_jsonfile("../source_datasets/", "gender_of_persons")

gender_of_persons = []
already_seen = set() #to remove duplicates
for person in gender_of_persons_original:
    if person["word"] not in already_seen:
        already_seen.add(person["word"])
        gender_of_persons.append(person)

#a list of dictionaries, one dictionary for each word
#a dictionary with keys "word", "wordnet_senseno" (useless), "gender" (str 'm', 'f', 'n' or 'o') and "gender_map"
#gender map is a dictionary whose keys are genders and values a list with 1 element being a dictionary whose entry
#"word" gives the quivalent word for the considered gender
#So, for an element which is male, elt["gender_map"]["f"][0]["word"] gives the female equivalent

gender_of_persons_simple = [{'word': person['word'], 'gender': person['gender']} for person in gender_of_persons]
#We have removed the mapping and other useless keys (useless except for statistics)

store_obj_in_jsonfile(gender_of_persons_simple, '../created_datasets/utils_datasets/', "clean_gender_of_persons")

#Creation of gendered_persons
# ! We lower them, because the contexts will be lowered to be processed
gendered_persons = []
for person in gender_of_persons_simple:
    if person['gender'] in 'mfo' and "_" not in person["word"]:
        #We add the person :
        person = person['word']
        gendered_persons.append(person)
        #We add its plural :
        gendered_persons.append(plural_form(person))

store_obj_in_jsonfile(gendered_persons, '../created_datasets/utils_datasets/', "gendered_persons")

#The special case of expressions consisting in several words (ie. with underscores in the original dataset)
gendered_underscores = [person["word"] for person in gender_of_persons
                        if "_" in person["word"] and person["gender"] in "mfo"]
trivial_gendered_persons = ["man", "woman", "boy", "girl", "lady", "brother", "sister", "father", "mother",
                            "male", "female", "son", "daughter", "king", "queen"]
#These trivial gendered persons are already included in gendered_persons
several_words_gendered_persons = [] #without the trivial ones
for expression in gendered_underscores:
    if not np.any([gendered_word in expression for gendered_word in trivial_gendered_persons]):
        # We add the person :
        person = " ".join(expression.split("_"))
        several_words_gendered_persons.append(person)
        # We add its plural :
        # As some of them are nouns, plurals don't mean anything,
        # but they shouldn't discard any correct example neither
        several_words_gendered_persons.append(plural_form(person))

store_obj_in_jsonfile(several_words_gendered_persons, '../created_datasets/utils_datasets/',
                      "several_words_gendered_persons")
# At the end, I will not use it during dataset creation, because most of them are irrelevant, that is to say
#already included in single words, and sometimes false
# (for instance 'sea scout', 'peer of the realm', 'taxi dancer' are gendered)
#The only ones that are interesting are 'lounge lizard' 'sex kitten', 'sumo wrestler', 'wet nurse'
# ('white slave' may refer to 'white slave trade affair', which is female gendered)

#STATISTICS

#To see which words are in the category other
others = [person["word"] for person in gender_of_persons if person["gender"]=="o"]
print(f"others:{others}\n") #hermaphrodite

#To enable easy mapping :
only_gender_of_persons = list(person["gender"] for person in gender_of_persons)
only_persons = list(person["word"] for person in gender_of_persons)

nb_persons = len(only_gender_of_persons)
nb_male = only_gender_of_persons.count("m")
nb_female = only_gender_of_persons.count("f")
nb_neutral = only_gender_of_persons.count("n")
nb_other = only_gender_of_persons.count("o")

nb_female_with_male_equivalent = 0
nb_male_with_female_equivalent = 0
nb_gendered_with_neutral_equivalent = 0
nb_female_with_male_equivalent_neutral = 0
nb_male_with_female_equivalent_neutral = 0

for person in gender_of_persons:
    if person["gender"] in "mf":
        if "gender_map" in person.keys():
            if "n" in person["gender_map"].keys():
                nb_gendered_with_neutral_equivalent += 1
                #print(f"gendered word with neutral equivalent : {person['word']}") #fireman (firefighter)
            if "m" in person["gender_map"].keys():
                #Then necessarily person["gender"] is "f"
                nb_female_with_male_equivalent += 1

                male_equivalent = person["gender_map"]["m"][0]["word"]
                #Now we look for this word in the dataset and look at his gender
                if male_equivalent in only_persons:
                    gender_male_equivalent = only_gender_of_persons[only_persons.index(male_equivalent)]
                    if gender_male_equivalent == "n":
                        nb_female_with_male_equivalent_neutral += 1
                        #print(f"{person['word']}, {male_equivalent}")

            elif "f" in person["gender_map"].keys():
                nb_male_with_female_equivalent += 1

                female_equivalent = person["gender_map"]["f"][0]["word"]
                # Now we look for this word in the dataset and look at his gender
                if female_equivalent in only_persons:
                    gender_female_equivalent = only_gender_of_persons[only_persons.index(female_equivalent)]
                    if gender_female_equivalent == "n":
                        nb_male_with_female_equivalent_neutral += 1
                        #print(f"{person['word']}, {female_equivalent}")

statistics = {'nb_persons': nb_persons, 'nb_male': nb_male, 'nb_female': nb_female, 'nb_neutral': nb_neutral,
              'nb_other': nb_other, 'nb_female_with_male_equivalent': nb_female_with_male_equivalent,
              'nb_male_with_female_equivalent': nb_male_with_female_equivalent,
              'nb_gendered_with_neutral_equivalent': nb_gendered_with_neutral_equivalent,
              'nb_female_with_male_equivalent_neutral': nb_female_with_male_equivalent_neutral,
              'nb_male_with_female_equivalent_neutral': nb_male_with_female_equivalent_neutral}

store_obj_in_jsonfile(statistics, '../saved_objects/context_dataset_creation/created_statistics/utils_statistics/',
                      "statistics_gender_of_persons")
