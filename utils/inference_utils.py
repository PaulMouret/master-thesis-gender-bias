import huggingface_hub
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from datasets import load_dataset
import time
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
import math
from copy import deepcopy

import sys

from utils.global_utils import relevance_score, bias_score
from utils.perplexity import compute_perplexity
from dama_utils.dama_l_hparams import DAMALeaceHyperParams
from dama_utils.model_utils import load_dama_model
from utils.constants_utils import *
from utils.dama_based_inference_utils import *


if torch.cuda.is_available():
    torch.set_default_device('cuda')


# huggingface_hub.login(READING_TOKEN)
# may be necessary in a first time in order to download HF models
# Once the model is saved locally, this command is not necessary anymore

# 0. LOADING MODEL AND TOKENIZER

def load_model(model_name):
    """
    Loads a model and returns it

    Parameters
    -----------
    model_name: str
        the name of the model, as specified on HuggingFace

    Returns
    ___________
    model:
        the model
    """
    if "peft" in model_name and not "merged" in model_name: #"finetuned_models"
        # it means it is a (HF) Peft model
        #print("We load a Peft model")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME,
                                                     torch_dtype=torch.float16 if torch.cuda.is_available()
                                                     else torch.float32,
                                                     low_cpu_mem_usage=True,
                                                     device_map='auto'
                                                     )

        if " " in model_name:
            name_revision = model_name.split()
            #print(f"Revision detected : {name_revision[0]} ({name_revision[1]})")
            model = PeftModel.from_pretrained(model, name_revision[0], revision=name_revision[1])
        else:
            model = PeftModel.from_pretrained(model, model_name)

        #model = model.merge_and_unload() #sometimes the resulting model is identical to the original one : weird
        #print("Peft model merged")

        #HF code

        #config = PeftConfig.from_pretrained("PaulM2000/peft_dama_finetuning_random_42_without_up_proj_Llama-2-7b-hf")
        #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        #model = PeftModel.from_pretrained(model, "PaulM2000/peft_dama_finetuning_random_42_without_up_proj_Llama-2-7b-hf")

    elif "dama" in model_name:
        torch.set_default_dtype(torch.float16)  # so that dama works
        # from https://stackoverflow.com/questions/73530569/pytorch-matmul-runtimeerror-addmm-impl-cpu-not-implemented-for-half

        end_name = model_name.split("dama_model_")[1]
        start_layer = int(end_name.split("-")[0])
        end_layer = int(end_name.split("-")[1].split("_")[0])
        num_layers = end_layer - start_layer + 1
        train_dataset = end_name.split("Llama-2-7b-hf_")[1]

        projections_dir = os.path.join("../DAMA_L", train_dataset, f"start_layer_{start_layer}")
        print(f"projections_dir : {projections_dir}")
        projection_file = os.path.join(projections_dir, "projections.npy")
        hparams_dir = os.path.join("../DAMA_L", train_dataset, f"start_layer_{start_layer}", f"{num_layers}L")
        print(f"hparams_dir : {hparams_dir}")
        hparams = DAMALeaceHyperParams.from_json(os.path.join(hparams_dir, "hparams.json"))

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME,
                                                     torch_dtype=torch.float16 if torch.cuda.is_available()
                                                     else torch.float32,
                                                     low_cpu_mem_usage=True, device_map='auto'
                                                     )

        model = load_dama_model(model, hparams, projection_file)

    elif "merged" in model_name:
        #it means it is a (HF) merged model

        if " " in model_name:
            name_revision = model_name.split()
            model = AutoModelForCausalLM.from_pretrained(name_revision[0], revision=name_revision[1],
                                                         torch_dtype=torch.float16 if torch.cuda.is_available()
                                                         else torch.float32,
                                                         low_cpu_mem_usage=True, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         torch_dtype=torch.float16 if torch.cuda.is_available()
                                                         else torch.float32,
                                                         low_cpu_mem_usage=True, device_map='auto')

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float16 if torch.cuda.is_available()
                                                     else torch.float32,
                                                     low_cpu_mem_usage=True, device_map='auto')

    return model


def load_tokenizer(model_name, pad_token="unk", pad_side="left"):
    """
    Loads a tokenizer, manages its special tokens and returns it

    Parameters
    -----------
    model_name: str
        the name of the model corresponding to the tokenizer, as specified on HuggingFace
    pad_token: str
        Indicates according to which policy the padding token should be defined
        choose among "bos", "eos", "pad"

    Returns
    ___________
    tokenizer:
        the tokenizer
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, return_token_type_ids=False, add_bos_token=False)
    #For some strange reason, if we add the bos token at the beginning
    # (even if the padding token is different from bos token) we get very bad results (ie. very low probabilities, e-10)
    # and very long inference time
    # set llama special tokens, inspired from DAMA code
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"
    tok.padding_side = pad_side
    #padding on the left, so the the last token is the token of interest regarding predictions

    if pad_token == "bos":
        tok.pad_token = tok.bos_token
    elif pad_token == "eos":
        tok.pad_token = tok.eos_token
    elif pad_token == "unk":
        tok.pad_token = tok.unk_token
    elif pad_token == "pad":
        tok.pad_token = "[PAD]"
    else:
        raise Exception(f"Incorrect value for the argument pad_token : {pad_token}")
    #The padding token shouldn't matter becasue of the attention masks,
    # I chose bos by default in the past because it's the one that would give the best results without attention mask
    # the id is 1 for bos, 2 for eos and 0 for <unk>
    # But in DAMA code (and in particular the function make_inputs) the pad toke id is considered to be 0,
    # so in order to match I choose the unk token now

    return tok


# 1. INFERENCE STEPS


def tokenize(strings, tokenizer, padding=True):
    """
    Returns the tokenized version of the input strings

    Parameters
    -----------
    strings: str or List[str]
        the set of strings we want to tokenize
    tokenizer:
        the tokenizer we want to use
    padding: bool
        indicates if we preliminary pad the strings
        Note that, if strings contains several strings (with a different length in terms of tokens),
        padding should be set to True, or it would raise an error

    Returns
    ___________
    tokenized_inputs: BatchEncoding
        tokenized version of the input strings
    """
    #Old version
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #return tokenizer(strings, return_tensors="pt", padding=padding, return_token_type_ids=False).to(device)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    token_lists = [tokenizer.encode(p) for p in strings]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def get_logits(batch_encoding_input, model):
    """
    Get the logits returned by some model for some input, in inference mode

    Parameters
    -----------
    batch_encoding_input: BatchEncoding
        (already tokenized) input
    model:
        the model we want to use

    Returns
    ___________
    logits: tensor
        the logits returned by the model
    """
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        logits1 = model(**batch_encoding_input)
        logits = logits1.logits
        #For some strange reason, when put together in a line,
        # this could cause an error when calling naive_full_inference
    return logits


def get_pronouns_indices(pronouns, tokenizer):
    """
    Get the tokenization ID's of some pronouns
    As pronouns are short, the expected tokenization for one pronoun is 1

    Parameters
    -----------
    pronouns: List[str]
        a list of pronouns
        Note that, depending on the model, we may want to add a space " " before
    tokenizer:
        the tokenizer we want to use

    Returns
    ___________
    indices: tensor
        a 1-D tensor whose each element correspond to the ID of the corresponding pronoun in pronouns
    """
    indices = tokenize(pronouns, tokenizer)["input_ids"].reshape((-1,))
    return indices


def get_pronouns_probabilities(logits, pronouns_indices):
    """
    From a full logits tensor, returns the probabilities corresponding to some specific tokens

    Parameters
    -----------
    logits: tensor
        a full logit tensor
    pronouns_indices: tensor
        the token ID's of the pronouns of interest, corresponding to the logits of interest

    Returns
    ___________
    pron_prob: tensor
        the tensor containing only the probabilities corresponding to pronouns_indices
    """
    probabilities = torch.softmax(logits[:, -1, :], dim=1)
    #proababilities has shape (nb_sentences, vocab_size) : it represents probabilities for the last token
    pron_prob = probabilities[:, pronouns_indices]
    return pron_prob


def get_top_one_predictions(logits, tokenizer):
    """
    From a full logits tensor, returns the top one prediction (as a string) for each element of the batch

    Parameters
    -----------
    logits: tensor
        a full logit tensor
    tokenizer:
        the tokenizer we want to use
        (it should correspond to the same model from which logits were computed)

    Returns
    ___________
    most_probable_words: List[str]
        the most probable last token, for each element of the batch
    """
    probabilities = torch.softmax(logits[:, -1, :], dim=1)
    # proababilities has shape (nb_sentences, vocab_size) : it represents probabilities for the last token
    most_probable = torch.argmax(probabilities, dim=1)
    biggest_probabilities = torch.max(probabilities, dim=1)
    most_probable_words = tokenizer.batch_decode(most_probable)
    return biggest_probabilities, most_probable_words


def get_top_k_predictions(logits, tokenizer, k=10):
    probabilities = torch.softmax(logits[:, -1, :], dim=1)
    #print(f"probabilities.shape : {probabilities.size()}")
    # proababilities has shape (nb_sentences, vocab_size) : it represents probabilities for the last token
    most_probable = torch.argsort(probabilities, dim=1, descending=True)[:, :k]
    #print(f"most_probable.shape : {most_probable.size()}")
    biggest_probabilities, indices = torch.sort(probabilities, dim=1, descending=True)
    biggest_probabilities = biggest_probabilities[:, :k]
    #print(f"biggest_probabilities.shape : {biggest_probabilities.size()}")
    #print(f"most_probable :\n{most_probable}")
    nb_sentences, k = most_probable.size()
    most_probable_words = [[tokenizer.decode(most_probable[i, j]) for j in range(k)] for i in range(nb_sentences)]
    biggest_probabilities = biggest_probabilities.tolist()

    return biggest_probabilities, most_probable_words


def get_top_k_predictions_from_strings(strings, model_name, padding=True, k=10):
    model = load_model(model_name)
    tokenizer = load_tokenizer(BASE_MODEL_NAME)
    batch_encoding_input = tokenize(strings, tokenizer, padding)
    logits = get_logits(batch_encoding_input, model)
    biggest_probabilities, most_probable_words = get_top_k_predictions(logits, tokenizer, k)
    return biggest_probabilities, most_probable_words


def get_pronouns_probabilities_from_strings(strings, model_name, padding=True,
                                            get_predictions=False, verbose=False):
    """
    Returns, for the Llama model, the pronouns probabilities corresponding to some input strings

    Parameters
    -----------
    strings: List[str]
        a set of input strings
    model_name: str
        the name of the model (for which there also exists a corresponding tokenizer) to use
    padding: bool
        indicatesif we pad the sentences
    get_predictions: bool
        indicates if the function also prints the top one predictions
    verbose: bool
        indicates the verbosity of the function

    Returns
    ___________
    pronouns_probabilities: tensor
        the tensor containing only the probabilities corresponding to pronouns_indices
    """
    #Note that if there are several strings in strings, padding should be true :
    #else the embeddings wouldn"t be rectangular, which would raise an exception (when creating the embeddings)
    model = load_model(model_name)
    tokenizer = load_tokenizer(BASE_MODEL_NAME)

    pronouns = ['he', 'she']
    pronouns_indices = get_pronouns_indices(pronouns, tokenizer)

    # Running the model
    batch_encoding_input = tokenize(strings, tokenizer, padding)
    if verbose:
        print(f"batch encoding inputs :\n{batch_encoding_input}")
    logits = get_logits(batch_encoding_input, model)

    # Extracting the information we want
    pronouns_probabilities = get_pronouns_probabilities(logits, pronouns_indices)

    if verbose:
        print(f"pronouns_probabilities.size() : {pronouns_probabilities.size()}")
        print(f"pronouns_probabilities : {pronouns_probabilities}")

    if get_predictions:
        top_one_probs, top_one_predictions = get_top_one_predictions(logits, tokenizer)
        print("top_one_predictions : ", top_one_predictions)
        print("top_one_probs : ", top_one_probs)

    return pronouns_probabilities


def full_inference(context_dataset_name, subset_size, batch_size, model_name="meta-llama/Llama-2-7b-hf",
                   random_seed=42, verbose=False):
    #Note that it works properly on GPU, ie. running it on GPU is indeed significantly faster
    start = time.time()
    res = []

    my_dataset = load_dataset('json', data_files=f'../created_datasets/{context_dataset_name}.jsonl',
                              split='train')
    if subset_size == 0:
        subset = my_dataset
    else:
        subset = my_dataset.train_test_split(test_size=min(subset_size, my_dataset.shape[0] - 1),
                                             seed=random_seed)['test']
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)

    #We load the model and the tokenizer, in order not to call them everytime
    model = load_model(model_name)
    tokenizer = load_tokenizer(BASE_MODEL_NAME)  # because no tokenizer directly associated to peft adapters
    pronouns = ['he', 'she']
    pronouns_indices = get_pronouns_indices(pronouns, tokenizer)
    loading = time.time()

    for batch in dataloader:
        batch_encoding_input = tokenize(batch['text'], tokenizer, padding=True)
        logits = get_logits(batch_encoding_input, model)
        pronouns_probabilities = get_pronouns_probabilities(logits, pronouns_indices).tolist()
        res += pronouns_probabilities
    end = time.time()
    loading_time = loading - start
    inference_time = end - loading

    return res, loading_time, inference_time


def naive_full_inference(context_dataset_name, subset_size, batch_size, model_name="meta-llama/Llama-2-7b-hf",
                         random_seed=42, verbose=False):
    #Should not be used, because it is not optimal compared to full_inference
    # (here the model is loaded several times)

    start = time.time()
    res = []
    my_dataset = load_dataset('json', data_files=f'../created_datasets/{context_dataset_name}.jsonl',
                              split='train')
    if subset_size == 0:
        subset = my_dataset
    else:
        subset = my_dataset.train_test_split(test_size=min(subset_size, my_dataset.shape[0] - 1),
                                             seed=random_seed)['test']
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    loading = time.time()

    for batch in dataloader:
        pronouns_probabilities = get_pronouns_probabilities_from_strings(batch['text'], padding=True,
                                                                         model_name=model_name).tolist()
        res += pronouns_probabilities

    end = time.time()
    loading_time = loading - start
    inference_time = end - loading

    return res, loading_time, inference_time


def relevance_and_bias_mapping_function(example):
    #Once we have the probabilities for 'he' and 'she', it is easy to compute all these values ;
    #the simplest seems to store it directly in the dataset instead of computing it everytime
    relevance = relevance_score(example["prob_he"], example["prob_she"])
    bias = bias_score(example["prob_he"], example["prob_she"])
    male_stereotype = bias > 0
    return {"relevance_score": relevance, "bias_score": bias, "male_stereotype": male_stereotype}


#In the following, I shouldn't have to create datasets corresponding to specific relevance scores :
#I just have to use the filter method of datasets
def add_inference_to_subset(context_dataset_name, subset_size, batch_size,
                            model_name,
                            random_seed=42, verbose=False):
    if torch.cuda.is_available():
        str_device = ''
    else:
        str_device = "_cpu"

    start = time.time()
    #The beginning is similar to full_inference
    #But we don't call full_inference because we need the subset
    res = []
    my_dataset = load_dataset('json', data_files=f'../created_datasets/{context_dataset_name}.jsonl',
                              split='train')
    if verbose:
        print("Dataset loaded")

    # Depending on the dataset, it may already have been inferred ; and more importantly,
    # it may have attributes whose length varies from a sentence to another, perturbating the dataloader. So :
    potential_columns_to_remove = ["corrupted_probs_he", "corrupted_probs_she",
                                   "corrupted_relevances", "corrupted_biases",
                                   "bias_differences", "relevance_differences", "related_words",
                                   "prob_he", "prob_she", "relevance_score", "bias_score", "male_stereotype",
                                   "parsed_original_text"]
    for colname in potential_columns_to_remove:
        if colname in my_dataset.column_names:
            my_dataset = my_dataset.remove_columns(colname)
    if verbose:
        print(f"remaining column names in my_dataset : {my_dataset.column_names}")

    if subset_size == 0:
        subset_size_used = len(my_dataset)
    else:
        subset_size_used = subset_size

    my_dataset = my_dataset.shuffle(seed=random_seed)
    subset = my_dataset.filter(lambda x, i: i < subset_size_used, with_indices=True)

    # In the past I encountered problems of identical inference after finetuning ; by using subset = my_dataset
    # deepcopy does not help
    if verbose:
        print(f"Subset of size {subset_size_used} created")

    tokenizer = load_tokenizer(BASE_MODEL_NAME)
    model = load_model(model_name)

    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)

    for batch in dataloader:
        pronouns_probabilities = dama_inference(model, tokenizer, batch['text'])
        #if verbose:
            #print(f"pronouns_probabilities :\n{pronouns_probabilities}")
        list_probs = pronouns_probabilities.cpu().tolist()
        res += list_probs

    #Now we calculate the relevances
    res = np.array(res)
    res_he = res[:, 0].squeeze()
    res_she = res[:, 1].squeeze()

    if verbose:
        for i in range(10):
            print(f"{subset['text'][i]} ({res_he[i]}, {res_she[i]})")

    subset = subset.add_column("prob_he", res_he)
    subset = subset.add_column("prob_she", res_she)
    subset = subset.filter(lambda x: x['prob_he'] != 0 and x['prob_she'] != 0)
    #In a subset of size 50000, there was 1 example with 0 probabilities : it was some URL code line

    subset = subset.map(relevance_and_bias_mapping_function)

    #For some strange reason, over 10.000 examples, 2 of them had None probabilities (4 over 20.000)
    #'thankfully my sword had been enchanted to be extremely sharp as well as return to my hand if'
    #'my bedroom is about the size of a closet and contains a shitty mattress , a box with stuff that'
    #'it was in the intricate cave system of this mountain range that they managed to carve out dungeons that'
    #'riffletic ,'
    # For debugging
    exceptions = subset.filter(lambda x: x["relevance_score"] is None or np.isnan(x["relevance_score"]))
    print(f"{len(exceptions)} contexts (over {len(subset)}) have nan inference (and have been removed) :")
    for exception in exceptions:
        print(exception["text"])

    subset = subset.filter(lambda x: x["relevance_score"] is not None and not np.isnan(x["relevance_score"]))

    jsonl_name = f"inferred_{context_dataset_name.split('/')[-1]}_{subset_size}_{random_seed}_" \
                 f"{'_'.join(model_name.split('/')[-1].split())}" + str_device
    # The .join is necessary for finetuned models where space in the name ;
    # but it shouldn't change anything compared to old code for base model

    subset.to_json(f"../created_datasets/inferred_datasets/{jsonl_name}.jsonl")

    end = time.time()
    total_time = end - start
    if verbose:
        print(f"{total_time} s for {subset_size_used} contexts -> "
              f"average time per context : {round(total_time/subset_size_used,3)} s.")

    #We delete variable to avoid GPU memory problems
    del model
    del res
    del my_dataset
    del subset
    del dataloader


def get_perpexity_from_names(model_name, dataset_name=f"../source_datasets/wikitext_test.jsonl",
                             pad_side="left"):
    #inspired from evaluate() in causal_lm.py in DAMA code
    model = load_model(model_name)
    base_tokenizer = load_tokenizer(BASE_MODEL_NAME, pad_side=pad_side)
    my_dataset = load_dataset('json', data_files=dataset_name,
                              split='train').to_iterable_dataset()
    results = compute_perplexity([te['text'] for te in my_dataset if len(te['text']) > 0],
                                 model, base_tokenizer)
    return results['mean_perplexity']


def print_parameters_of_model(model_name, l=21):
    model = load_model(model_name)

    for name, param in model.named_parameters():
        if f"{l}.mlp.down_proj" in name:
            print(name)
            a = param.data[:5, :5]
            print(a.cpu().numpy())
