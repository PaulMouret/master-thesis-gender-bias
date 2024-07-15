import sys

from utils.global_utils import load_obj_from_jsonfile


books = load_obj_from_jsonfile("../source_datasets/", "books")

for book in books['books']:
    if 'id' not in book.keys() or 'language' not in book.keys():
        print(f"exception : {book}")
    else:
        if book['language'] != "English":
            print(f"{book['id']} : {book['language']}")

#magazine have no key 'language", but they turn out to be in English when checking manually
#We removed the books that were not in English (for instance Jules Verne books in French) before creating all_sf.txt
