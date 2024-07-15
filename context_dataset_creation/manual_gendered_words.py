import sys

from utils.global_utils import store_obj_in_jsonfile, plural_form


manual_male_gendered_persons = ['daddy', 'papa', 'grandpa', 'great-grandpapa', 'stepdad', 'grand-pere',
                                'guy', 'jax', 'chap', 'lad', 'bridegroom', 'sissy', 'prince', 'dude', 'bloke',
                                'bro', 'granddad', 'mister', 'brother_', 'gramp']
manual_male_non_plural = ['mr', '.he', '_he']
manual_male_firstnames = ['Nico', 'Jules', 'Alcide', 'Boris', 'Daemon', 'Ari', 'Rafe', 'Jase', 'Nash',
                          'Garion', 'Ryker', 'Teeg', 'Blackbeard', 'Drax', 'Aristotle', 'Kaladin', 'Belgarath', 'Zeus',
                          'Atreus', 'Blay', 'Caine', 'Brandt', 'Diesel', 'Bates', 'Harlin', 'Hauk', 'Smith', 'Izzy',
                          'Rome', 'Luk', 'Satan', 'Buddha', 'Obama', 'Yahweh', 'Hitler', 'Trump',  'Achilles',
                          'Lucifer', 'Vishnu', 'Sherlock', 'Loki', 'Allah', 'Reese', 'Shem', 'Ridmark', 'Ishmael',
                          'Agamemnon', 'Nietzsche', 'Judas', 'Stalin', 'Marx', 'Einstein', 'Rasûlullah', 'McCarthy',
                          'Mykael', 'Beethoven', 'Nixon', 'Socrates', 'Alfie', 'Odin', 'Lazarus', 'Thorik', 'Ryson',
                          'Ranmaru', 'Odysseus', 'Olórin', 'Odisseus', 'Niccolo', 'Seifer', 'Aleric', 'Hercules',
                          'Falignus', 'Arien', 'Eli', 'Gramps']

manual_female_gendered_persons = ['mommy', 'mama', 'mom', 'gran', 'mum', 'grandma', 'grandmother',
                                  'stepmom', 'momma', 'stepsister', 'step-daughter', 'grand-mere', 'auntie', 'wench',
                                  "mistress", 'goddess', 'bitch', 'chantress', 'witch', 'maiden', 'majorette',
                                  'aunty', 'sis', 'mummy', 'chick', 'maman', 'mademoiselle', '_mom', 'wives']
manual_female_non_plural = ['mrs', 'ms', '.she', '_she', '_she_']
manual_female_firstnames = ['Jaenelle', 'Sofia', 'Valentina', 'Natalie', 'Jessica', 'Maysie', 'Diandra',
                            'Julieth', 'Adalynn', 'Tatianna', 'Reah', 'Clary', 'Dee-Dee', 'Ana', 'Kimber', 'Flaevynn',
                            'Sookie', 'Polgara', "Ce'nedra", 'Rapunzel', 'Bea', 'Gabby', 'Islena', 'Paige',
                            'Zhaan', 'Kiya', 'Mahrree', 'Liv', 'Rei', 'Scarlet', 'Persephone', 'Aerin', 'Aphrodite',
                            'Bri', 'Kriista', 'Elowyn', 'Eirwen', 'Gennie', 'Kyrah', 'Serinae', 'Aradia', 'Alissia',
                            'Lyssa', 'Aya', 'Ela', 'Rhetta', 'Abbi', 'Akilah', 'Electra', 'Zoë', 'Mirra', 'Ceecee',
                            'Cynara', 'Morrigan', 'Kyrianna', 'Taysha', 'Katryn', 'Sera', 'Azalea', 'Trula',
                            'Channie', 'Kiya', 'Brie', 'Lark', 'Skylar', 'Mirana', 'Alanki', 'Thetra', 'Abra',
                            'Miako', 'Elliana', 'Ceecee', 'Calliande', 'Middleton', 'Jacy', 'Jennadil',
                            'Kyrin', 'Quen', 'Sadey', 'Celestia', 'Lia', 'Mags', 'Myrhia', 'Tika', 'Suella',
                            'Meeka', 'Gersina', 'Tink', 'Ajita', 'Clarity', 'Mercy']


#Because we check the gendered worsd on the lowered sentence, we store the lowercase gendered words

#Some common gender-neutral (but stereotyped) first names that should not be added :
#Morgaine, Sam, Saffi, Dominique
#Some first names are so rare that they may appear only in one given book ; e.g. see 'flaevynn' on Google
#(idem for belgarath, polgara, "ce'nedra") : it's almost surprising that they are gendered correctly

#rafe is the short from rafael, which is male gendered
# hed actually stands for he'd, that for some reason appears this way in bookcorpus ; idem for shed

manual_gendered_persons = manual_male_gendered_persons + manual_female_gendered_persons
firstnames = manual_male_firstnames + manual_female_firstnames
plurals = []

for person in manual_gendered_persons:
    plurals.append(plural_form(person))

store_obj_in_jsonfile(manual_gendered_persons + plurals + manual_male_non_plural + manual_female_non_plural,
                      '../created_datasets/utils_datasets/', "manual_gendered_persons")

store_obj_in_jsonfile(firstnames,
                      '../created_datasets/utils_datasets/', "manual_gendered_firstnames")
