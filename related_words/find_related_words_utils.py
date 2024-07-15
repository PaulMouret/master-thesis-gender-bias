from datasets import load_dataset
from blingfire import text_to_words
from spacy.tokens import Doc
from spacy import load
from fastcoref import FCoref
import re
from nltk import Tree
import numpy as np
from spacy_conll import init_parser
from spacy_conll.parser import ConllParser


# We build the nlp object : actually it is just a converter in our case
nlp = ConllParser(init_parser("en_core_web_sm", "spacy"))


#to print the trees for debugging
def tok_format(tok):
    return "_".join([tok.orth_, str(tok.i), tok.dep_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)


#A util function
def recursive_children_set(token):
    new_set = set()
    for child in token.children:
        #print(f"child is {child.text} and its i is {child.i}")
        new_set.add(child.i)
        children_children = recursive_children_set(child)
        #print(f"Its own children are {children_children}")
        new_set = new_set.union(children_children)
    return new_set


# 1. We create the function to map
def related_words_mapping_function(example, verbose=False):
    if verbose:
        print(f"\n###############\n{example['original_text']}")
    #For debugging
    #for word in {"he", "she", "the", "patient"}:
        #occurences_in_sentence = [m.start() for m in re.finditer(word, example["original_text"])]
        #print(f"occurrences of {word} : {occurences_in_sentence}")
    # a. We find the targets
    # ! In practice the coreference solver does not work well for dialogue contexts but we can implement it manually
    if example["dialogue_context"]:
        start_i = []
        for i_word, word in enumerate(text_to_words(example['text']).lower().split()):
            if word in {"i", "me", "my", "mine", "we", "us", "our", "ours"}:
                start_i.append(i_word)
        if verbose:
            if start_i:
                print(f"target cluster : {np.array(text_to_words(example['text']).split())[np.array(start_i)]}")
            else:
                print("target cluster : []")

    else:
        model = FCoref()
        preds = model.predict(texts=[example['original_text']])
        index_pronoun = len(example['text']) + 1
        #print(f"index_pronoun is {index_pronoun}")
        # We find the start indices of the pronoun and the words that refer to it
        start_indices = []
        clusters = preds[0].get_clusters(as_strings=False)
        if verbose:
            print(f"clusters : {clusters}")
        for i_cluster, cluster in enumerate(clusters):
            if not None in cluster:
                #somehow in example "One who thinks before he speaks," a mention of one cluster was None
                for (start, end) in cluster:
                    if start == index_pronoun: #it is the target cluster
                        strings_cluster = preds[0].get_clusters()[i_cluster]
                        if verbose:
                            print(f"target cluster : {strings_cluster}")
                        for i_reference, reference in enumerate(strings_cluster):
                            start_ref = cluster[i_reference][0]
                            for word_of_reference in reference.split(): #because a cluster may cover several words
                                start_indices.append(start_ref)
                                start_ref += len(word_of_reference) + 1
            else:
                print(f"Obtain a None mention in a cluster for the following original text :"
                      f"\n{example['original_text']}")
        # print(f"start indices : {start_indices}")
        # start_indices may contain only index_pronoun, if no cluster is detected
        # We convert the start_indices into start_i (token indices instead of char indices), because .idx does not work correctly
        # print(f"the starts are {[text_to_words(example['text'][:start_index]).split() for start_index in start_indices]}")
        start_i = [len(text_to_words(example['text'][:start_index]).split()) for start_index in start_indices]

    # b. We analyze the parsing (whose results are already computed in doc)
    doc = nlp.parse_conll_text_as_spacy(example["parsed_original_text"])
    if verbose:
        [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

    #print(f"start_i : {start_i}")
    # We identify the token indices of related words
    related_token_indices = set()
    for token in doc:
        #print(f"token : {token.text} ; i : {token.i}")
        if token.i in start_i: #if it is a target node
            #print(f"tag {token.tag_} pos {token.pos_}")
            # a. We add itself
            related_token_indices.add(token.i)
            if verbose:
                print(f"1 : {token.lemma_}")
            # b. We add its children that directly refer to it (so no grandchildren etc.)
            for child in token.children:
                if (child.dep_ == 'det') or ("mod" in child.dep_) or ("flat" in child.dep_):
                    related_token_indices.add(child.i)
                    if verbose:
                        print(f"2 : {child.lemma_}")
                #We treat the specific case of relative for which the target is the subject
                elif child.dep_ == 'acl:relcl':
                    if child.left_edge.lemma_ in {'who', 'that', 'which'} and child.left_edge.dep_ == "nsubj":
                        related_token_indices.add(child.i)
                        if verbose:
                            print(f"3 : {child.lemma_}")

            # c. For the rest, it depends on the sentence :
            # c.i. if it is a subject
            if token.dep_ == "nsubj":
                parent = token.head
                #If it is a classical verb, we add it, but not the corresponding object
                #if parent.pos_ == "VERB":
                    #related_token_indices.add(parent.i)
                    #print(f"4a : {parent.lemma_}")
                #Else the verb must be a copula ; we can add both the verb and what follows it
                # we also had the conjonctions if there are
                #else:
                related_token_indices.add(parent.i)
                if verbose:
                    print(f"4 (parent): {parent.lemma_}")
                siblings = parent.children
                for sibling in siblings:
                    if sibling.dep_ in {"cop", "conj"}:
                        related_token_indices.add(sibling.i)
                        if verbose:
                            print(f"4 (sibling): {sibling.lemma_}")
            # c.ii. if it is a possessive (typically for 's or pronouns)
            elif token.dep_ == "nmod:poss":
                related_token_indices.add(token.head.i)
                if verbose:
                    print(f"5 : {token.head.lemma_}")

    #print(f"related_token_indices : {related_token_indices}")
    #Now we construct the list from these indices, keeping only the ones included in context
    # (because we did the parsing on the full sentence)
    max_id_token = len(text_to_words(example['text']).split())
    #We create the boolean mask
    related_words = [True if i in related_token_indices else False for i in range(max_id_token)]
    # it has the same shape as the context has tokens
    if verbose:
        print(f"related words are {np.array(text_to_words(example['text']).split())[np.array(related_words)]}")

    return {"related_words": related_words}

#Remarks for future improvement, with examples to improve :
# Alone in an old rusty 4x4 GAZ car, he had almost reached the village, amazingly without being attacked once by the dreaded beasts.
#       'Alone' is a siblig of 'he', with dep_ advcl
# "\"Maybe it was just a hunter, taking a shot at what he thought was a deer,\" Seth said."
#       'Maybe' is a child of hunter, with dep_ advmod
# "**Damnations will shower on a person who does not rescue the victim although he is present at the scene.\"**"
#       'victim', 'although' are considered related (must be because of wrong cluster)
# When Pilentum was at the great Warley Model Train Show in Birmingham in 2018, he saw so many beautiful model train layouts that he did not know what to film.
#       'Show' is parent of Pilentum which is nsubj (with copula verb)
# "You say I didn't do anything wrong, but something caused you to leave,\" he says."
#       'You' is considered related (must be because of wrong cluster)
