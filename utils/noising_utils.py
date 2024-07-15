from datasets import load_dataset
import torch
import numpy as np
import time
import re #to manage regular expressions
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM


import utils.nethook as nethook #a util file from DAMA code, that itself comes from Menge et al. paper


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("torch.cuda.is_available() : ", torch.cuda.is_available()) #sometimes True and sometimes False...
#.get_device()is -1 when it is cpu, else index of gpu (starting from 0)

#The following functions are (or are inspired) from DAMA code

#A class corresponding to a couple model/tokenizer
#Some modifications to match what I did previously, indicated by #!
class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
            self,
            model_name=None,
            model=None,
            tokenizer=None,
            low_cpu_mem_usage=False,
            torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      use_fast=True, #!
                                                      return_token_type_ids=False, #!
                                                      add_bos_token=False)
            #!
            tokenizer.bos_token = "<s>"
            tokenizer.eos_token = "</s>"
            tokenizer.unk_token = "<unk>"
            tokenizer.pad_token = tokenizer.unk_token
            #so that the corresponding id is 0, to match his function make_inputs
            tokenizer.padding_side = "left"
        if model is None:
            assert model_name is not None
            #!
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         torch_dtype=torch_dtype,
                                                         low_cpu_mem_usage=True, device_map='auto')
            model.eval()

            nethook.set_requires_grad(False, model)

        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

    def get_input_representations(self, prompt, average=True):
        """
        Get the input representations for a given prompt
        :param prompt: textual input
        :param average: whether to average embeddings across all tokens
        :return: torch.Tensor of shape (embedding_size,) or (num_tokens, embedding_size)
        """
        input_ids = self.tokenizer.encode(prompt)

        # get embeddings at given token positions
        input_embeddings = self.model.get_input_embeddings().weight[input_ids, :]
        if average:
            input_embeddings = input_embeddings.mean(dim=0, keepdim=False)
        return input_embeddings


#To easily access leayers
def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    # Update for LLaMa
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


#To ge the pronouns probabilities
PRONOUNS = (' she', ' he', ' they')
PRONOUNS_LLAMA = ('he', 'she') #! #just to be consistent with my previous code


#Note that it is supposed to work with either a set of corrupted inputs (except the first one) or all identical inputs,
#so it is different from my function
def get_pronoun_probabilities(output, mt, is_batched=False):
    if is_batched:
        probabilities = torch.softmax(output[1:, -1, :], dim=1).mean(dim=0)
    else:
        probabilities = torch.softmax(output[:, -1, :], dim=1).mean(dim=0)

    if "llama" in mt.model.name_or_path.lower():
        pronoun_tokens = PRONOUNS_LLAMA
    else:
        pronoun_tokens = PRONOUNS

    pron_prob = []
    for pronoun in pronoun_tokens:
        pron_prob.append(probabilities[mt.tokenizer.encode(pronoun)][0])

    return torch.stack(pron_prob)


def pronoun_probs(mt, inp):
    out = mt.model(**inp)
    probs = get_pronoun_probabilities(out.logits, mt, is_batched=False)
    return probs


#To create input dicts from strings
def make_inputs(tokenizer, prompts, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device).long(),
        position_ids=torch.tensor(position_ids).to(device).long(),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
# The .long is for debugging, following advice from
#https://stackoverflow.com/questions/56360644/pytorch-runtimeerror-expected-tensor-for-argument-1-indices-to-have-scalar-t


#A variation ; used for determining sigma_t
def make_inputs_for_sigma(tokenizer, encoded_prompts, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    token_lists = [[e] for e in encoded_prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device).long(),
        position_ids=torch.tensor(position_ids).to(device).long(),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
# The .long is for debugging, following advice from
#https://stackoverflow.com/questions/56360644/pytorch-runtimeerror-expected-tensor-for-argument-1-indices-to-have-scalar-t


#A util function
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


#To find the indexes of the tokens corresponding to a given substring in a given token_array :
#returns tok_start, tok_end : like when indexing strings, tok_end is not included
#N.B. : returns an error if the substring is not in the sentence corresponding to the token_array
#Note that both string and substrings undergo a .join operation with NO SPACE
#N.B. : Actually I modifiied it so that the substring undergoes no .join :
#this has no consequence if the substring is a single word, which is the case so far, but else it may be problematic
def find_token_range(tokenizer, token_array, substring, char_loc):
    toks = decode_tokens(tokenizer, token_array)
    #print(f"{toks} ({len(toks)})")
    #whole_string = "".join(toks)
    #print("whole_string :", whole_string)
    #substring = "".join(decode_tokens(tokenizer, tokenizer.encode(substring)))
    #print("substring : ", substring)
    #char_loc_original = whole_string.index(substring)
    #char_lo = whole_string.index(substring)
    #print("char_lo : ", char_lo, " char loc original ", char_loc_original)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    #print(f"\ntoks : {toks}\nsubstring : {substring}")
    #print(f"({tok_start}, {tok_end})")
    return (tok_start, tok_end)


def find_token_range_from_strings(tokenizer, string, substring, char_loc):
    toks = decode_tokens(tokenizer, tokenizer.encode(string))
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


#To get the variance of embeddings
def collect_embedding_std(mt, subjects, verbose=False):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
            #t.output_size : torch.Size([1, 2, 4096]
            #print("t.output[0].size() : ", t.output[0].size())
            #Strangely enough, it takes value torch.Size([k, 4096]) for k=0,1,2,3, even if most of the time it is 1
    alldata = torch.cat(alldata)
    if verbose:
        print("alldata.size() : ", alldata.size())
        #alldata.size(): torch.Size([8, 4096]) : (nb_max_token * nb_subjects, len_embedding)
    noise_level = alldata.std().item()
    # the std is caclulated by reducing to all dimensions
    # in other words we assume all components of the embedding vectors behave similarly
    alldata = alldata.tolist() #because I prefer to return a list object I can manipulate
    return alldata, noise_level


#A variation of collect_embedding_std that enables me to manage well special tokens ; for determining sigma_t
def collect_embedding_std_for_sigma(mt, encoded_subjects, verbose=True):
    alldata = []
    for s in encoded_subjects:
        inp = make_inputs_for_sigma(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
            #t.output_size : torch.Size([1, 2, 4096]
            #print("t.output[0].size() : ", t.output[0].size())
            #Strangely enough, it takes value torch.Size([k, 4096]) for k=0,1,2,3, even if most of the time it is 1
    alldata = torch.cat(alldata)
    if verbose:
        print("alldata.size() : ", alldata.size())
        #alldata.size(): torch.Size([8, 4096]) : (nb_max_token * nb_subjects, len_embedding)
    noise_level = alldata.std().item()
    # the std is caclulated by reducing to all dimensions
    # in other words we assume all components of the embedding vectors behave similarly
    alldata = alldata.tolist() #because I prefer to return a list object I can manipulate
    return alldata, noise_level


#Returns the probabilities where states in states to patch are restored
# (so, if trace_to_patch is [], this corresponds to the corrupted run)
# inp corresponds to a set of inputs corresponding to the same context :
# the 0-th is the correct one and the others are noised variations
# there are several of them because, as noise is random, to get a consistent result we average their results
def trace_with_patch(
        mt,  # The model and tokenizer
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
        project_embeddings=None  # INLP projection matrix to project embeddings
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(mt.model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
            mt.model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
    ) as td:
        outputs_exp = mt.model(**inp)

    # We report softmax probabilities for three gendered probabilites:
    probs = get_pronoun_probabilities(outputs_exp.logits, mt, is_batched=True)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


# Based on calculate_hidden_flow from DAMA code
def calculate_base_and_corrupted_scores(
        mt,
        prompt,
        subject,
        subject_index,
        samples=10,
        noise=0.1,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        base_score = pronoun_probs(mt, inp) #matches my probabilities

    answer_t = None
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject, subject_index)
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise, uniform_noise=False
    )

    return base_score, low_score


def calculate_corrupted_scores(
        mt,
        prompt,
        subject,
        subject_index,
        samples=10,
        noise=0.1,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    answer_t = None
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject, subject_index)
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise, uniform_noise=False
    )

    return low_score


#A mapping_function may have been useful, but fails in practice because of memory
