# Implementation from with changes : https://github.com/tomlimi/DAMA

import os, re, sys, json
import torch, numpy
import math
import argparse
from tqdm import tqdm
from collections import defaultdict
import time

import sys

from utils.constants_utils import *
from causal_trace import ModelAndTokenizer, guess_subject, calculate_hidden_flow, collect_embedding_std
from knowns import KnownsDataset
from causal_tracing_globals import *


torch.set_grad_enabled(False)


def load_data(mt, data_path, name_dataset, noise_level=None):
    knowns = KnownsDataset(data_path, name_dataset)
    if not noise_level:
        noise_level = FINAL_FACTOR * FINAL_STD
        #noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
    return knowns, noise_level


def compute_results(
        mt,
        prompt,
        subject,
        subject_index,
        samples=10,
        noise=0.1,
        window=10,
        kind=None,
        modelname=None,
        savepdf=None,
        disable_mlp=False,
        disable_attn=False,
        project_embeddings=None
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, subject_index, samples=samples, noise=noise, window=window, kind=kind,
        disable_mlp=disable_mlp, disable_attn=disable_attn,
        project_embeddings=project_embeddings
    )
    
    result = {field: value for field, value in result.items() if field in ("scores", "low_score", "high_score")}
    result["prompt"] = prompt
    result["subject"] = subject
    
    return result


def compute_save_gender_effects(result_path, name_dataset, mt, knowns, noise_level=0.08, cap_examples=1000,
                                disable_mlp=False, disable_attn=False,param_number=None, inlp_projection=None):
    
    param_str = f"_{param_number}B" if param_number is not None else ''
    disable_str = f"{'_disable_mlp' if disable_mlp else ''}{'_disable_attn' if disable_attn else ''}"
    inlp_str = f"_inlp" if inlp_projection else ""
    
    if cap_examples == -1:
        result_file = os.path.join(result_path,f"results_{name_dataset}{param_str}{disable_str}{inlp_str}_all.jsonl")
        
    else:
        knowns.shuffle(seed=92)
        knowns = knowns[:cap_examples]
        result_file = os.path.join(result_path,f"results_{name_dataset}{param_str}{disable_str}{inlp_str}_{cap_examples}.jsonl")

    if inlp_projection:
        if not os.path.exists(os.path.join(result_path,inlp_projection)):
            raise ValueError(f"Projection file {inlp_projection} does not exist")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        npzfile = numpy.load(os.path.join(result_path,inlp_projection))
        project_embeddings = {}
        for arr_name in {'P', 'mu', 's'}:
            if arr_name not in npzfile:
                raise ValueError(f"Projection file {inlp_projection} does not contain array {arr_name}!")
            project_embeddings[arr_name] = torch.from_numpy(npzfile[arr_name]).type(torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    else:
        project_embeddings = None


    results = {}
    with open(result_file, "w") as f:
        for knowledge in tqdm(knowns):
            #We copy all information from knowledge
            results['prompt'] = knowledge['prompt']
            results['subject'] = knowledge['subject']
            results['subject_index'] = knowledge['subject_index']
            results['subject_start_end'] = knowledge['subject_start_end']
            results["subject_male_stereotyped"] = knowledge["subject_male_stereotyped"]
            results["subject_related"] = knowledge["subject_related"]
            results["dialogue_context"] = knowledge["dialogue_context"]
            results["prob_he"] = knowledge["prob_he"]
            results["prob_she"] = knowledge["prob_she"]
            results["relevance_score"] = knowledge["relevance_score"]
            results["bias_score"] = knowledge["bias_score"]
            results["male_stereotype"] = knowledge["male_stereotype"]

            for kind in [None, "mlp", "attn"]:
                out = compute_results(mt, knowledge["prompt"], knowledge["subject"], knowledge["subject_index"],
                                      noise=noise_level, kind=kind,
                                      disable_mlp=disable_mlp, disable_attn=disable_attn, project_embeddings=project_embeddings)
                # convert torch tensors to json serializable lists
                results[kind] = {k: v.tolist() for k, v in out.items() if k in ("scores", "low_score", "high_score")}
                # normally I shouldn't need "high_score" (because I already have "prob_he" and "prob_she",
                # but at least it enables debugging comparison
            f.write(json.dumps(results) + "\n")
            f.flush()
            

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_name_path", type=str, default="meta-llama/Llama-2-7b-hf")
    argparse.add_argument("--known_dataset", type=str, default=None)
    argparse.add_argument("--param_number", type=int, default=None)
    argparse.add_argument("--noise_level", type=float, default=0.06)
    argparse.add_argument("--cap_examples", type=int, default=1000)
    argparse.add_argument("--disable_mlp", action="store_true", default=False)
    argparse.add_argument("--disable_attn", action="store_true", default=False)
    argparse.add_argument("--inlp_projection", type=str, default=None)
    args = argparse.parse_args()

    start = time.time()

    model_name = args.model_name_path
    if model_name.endswith("llama"):
        if args.param_number in {7, 13, 30, 65}:
            model_name += f"_{args.param_number}B"

    # load model and split over multiple gpus if necessary
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                           low_cpu_mem_usage=False)
    knowns, noise_level = load_data(mt, DATA_DIR, args.known_dataset, args.noise_level)
    
    compute_save_gender_effects(RESULTS_DIR, args.known_dataset, mt, knowns, noise_level, args.cap_examples, args.disable_mlp, args.disable_attn,
                                param_number=args.param_number, inlp_projection=args.inlp_projection)

    end = time.time()
    print(f"{end - start} s for {args.cap_examples}")
