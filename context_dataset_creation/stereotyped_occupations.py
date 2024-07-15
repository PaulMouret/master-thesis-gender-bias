import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import store_obj_in_jsonfile, plural_form


#In this case we don't really need any analysis
for filename in ['female_occupations', 'male_occupations']:
    with open(f'../source_datasets/{filename}.txt', 'r') as f:
        temp = f.read().splitlines() #it is not necessarily lowercase and plurals are lacking
        final = []
        for person in temp:
            #person = person.lower()
            final.append(person)
            final.append(plural_form(person))

        print(len(final))
        store_obj_in_jsonfile(final, '../created_datasets/utils_datasets/', filename)
