import csv

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import store_obj_in_jsonfile

#Creates the datasets and statistics related to firstnames

#IMPORTATION AND DATASET CREATION

#Creation and preprocessing of firstnames
#Firstnames is a list of dictionaries, whose keys are 'firstname' and 'gender'

#We will only keep firstnames from countries of the Anglosphere !
#Which corresponds, in the dataset, to Great Britain, Ireland and U.S.A

#Simultaneously to creation, we will create countries statistics
nb_british = 0
nb_irish = 0
nb_american = 0

firstnames = [] #first names that are of length > 3 or English
eng_firstnames = [] # English first names
already_seen = set() #to remove duplicates, it there are
with open("../source_datasets/firstnames.csv", 'r', encoding="utf8") as data:
    inter_symbols = {' ', '-'}
    for i, line in enumerate(csv.reader(data, delimiter=';')):
        if i == 0:
            china_index = line.index("China")
            korea_index = line.index("Korea")

            gb_index = line.index("Great Britain")
            ireland_index = line.index("Ireland")
            us_index = line.index("U.S.A.")
        else:
            gender = line[1]
            firstname = line[0]

            chinese = (line[china_index] != '')
            korean = (line[korea_index] != '')
            british = (line[gb_index] != '')
            irish = (line[ireland_index] != '')
            american = (line[us_index] != '')

            if firstname not in already_seen:
                already_seen.add(firstname)
                #update countries statistics
                if british:
                    nb_british += 1
                if irish:
                    nb_irish += 1
                if american:
                    nb_american += 1

                #process gender
                if "F" in gender:
                    gender = "f"
                elif "M" in gender:
                    gender = "m"
                elif gender == "?":
                    gender = "n"
                else:
                    print(f"Unknown gender ({gender}) for {firstname} at line {i}")

                #preprocess firstname
                if "+" in firstname:
                    beginning, end = firstname.split("+")[0], firstname.split("+")[1]
                    for inter_symbol in inter_symbols:
                        f = beginning + inter_symbol + end
                        if len(f) > 3 or british or irish or american:
                            firstnames.append({'firstname': f, 'gender': gender})
                        if british or irish or american:
                            eng_firstnames.append({'firstname': f, 'gender': gender})
                    if chinese or korean:
                        f = beginning + end.lower()
                        if len(f) > 3 or british or irish or american:
                            firstnames.append({'firstname': f, 'gender': gender})
                        if british or irish or american:
                            eng_firstnames.append({'firstname': f, 'gender': gender})
                else:
                    if len(firstname) > 3 or british or irish or american:
                        firstnames.append({'firstname': firstname, 'gender': gender})
                    if british or irish or american:
                        eng_firstnames.append({'firstname': firstname, 'gender': gender})


store_obj_in_jsonfile(firstnames, '../created_datasets/utils_datasets/', "clean_firstnames")
store_obj_in_jsonfile(eng_firstnames, '../created_datasets/utils_datasets/', "clean_eng_firstnames")

#Creation of gendered_firstnames
# ! We DON'T lower them
# ! We DONT'T include the possessive
gendered_firstnames = [firstname['firstname'] for firstname in firstnames if firstname['gender'] in 'mf']
store_obj_in_jsonfile(gendered_firstnames,
                      '../created_datasets/utils_datasets/', "gendered_firstnames")

#Creation of gendered_eng_firstnames
gendered_eng_firstnames = [firstname['firstname'] for firstname in eng_firstnames if firstname['gender'] in 'mf']
store_obj_in_jsonfile(gendered_eng_firstnames,
                      '../created_datasets/utils_datasets/', "gendered_eng_firstnames")

#STATISTICS

#For firstnames

only_genders = list(person["gender"] for person in firstnames)

nb_persons = len(only_genders)
nb_male = only_genders.count("m")
nb_female = only_genders.count("f")
nb_neutral = only_genders.count("n")

statistics = {'nb_persons': nb_persons, 'nb_male': nb_male, 'nb_female': nb_female, 'nb_neutral': nb_neutral}

store_obj_in_jsonfile(statistics, '../saved_objects/context_dataset_creation/created_statistics/utils_statistics/',
                      "statistics_firstnames")

#For eng_firstnames

eng_only_genders = list(person["gender"] for person in eng_firstnames)

eng_nb_persons = len(eng_only_genders)
eng_nb_male = eng_only_genders.count("m")
eng_nb_female = eng_only_genders.count("f")
eng_nb_neutral = eng_only_genders.count("n")

eng_statistics = {'eng_nb_persons': eng_nb_persons, 'eng_nb_male': eng_nb_male, 'eng_nb_female': eng_nb_female,
                  'eng_nb_neutral': eng_nb_neutral,
                  'nb_british': nb_british, 'nb_irish': nb_irish, 'nb_american': nb_american}

store_obj_in_jsonfile(eng_statistics, '../saved_objects/context_dataset_creation/created_statistics/utils_statistics/',
                      "statistics_eng_firstnames")
