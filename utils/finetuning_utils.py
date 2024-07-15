#Standard imports
from datasets import load_dataset
#Additional imports according to inspirational notebook
import torch
from peft import LoraConfig, PeftModel, get_peft_model, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import transformers
import huggingface_hub
from trl import SFTTrainer
import bitsandbytes #not used directly, but should be in library for the code to work
import time

import sys

from utils.constants_utils import *


#Based on https://colab.research.google.com/drive/14xo6sj4dARk8lXZbOifHEn1f_70qNAwy?usp=sharing#scrollTo=v2huC6dMh5vE


def finetuning(base_model_name, finetuned_model_name, revision,
               train_dataset_name, test_dataset_name, num_epochs, learning_rate,
               batch_size, gradient_accumulation_steps,
               num_layers, include_up_proj=False, include_gate_proj=False,
               lora_r=16, lora_alpha=16, lora_dropout=0.,
               verbose=True):
    """
    lora_r = 16 #Lora attention dimension (the “rank”) ; 8 by default
    lora_alpha = 16 #The alpha parameter for Lora scaling ; 8 by default
    #In the original paper, it is said "As a result, we simply set α to the first r we try and do not tune it."
    lora_dropout = 0.0 #The dropout probability for Lora layers ; 0.0 by default
    """

    # 1. Model and tokenizer

    # 1.a. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                              use_fast=True,
                                              return_token_type_ids=False,
                                              add_bos_token=False)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # 1.b. Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map='auto',
    )
    #Note that the loading of the model is different from my loading of the model in the rest of the code :
    #using float16 raises errors, and it is not possible to avercome them because of memory issues

    base_model.config.use_cache = False #recommended for finetuning

    if verbose:
        print("Model and tokenizer loaded")

    # 2. Peft Model

    #How to get the names of the modules :
    #https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models
    #https://huggingface.co/docs/peft/task_guides/semantic_segmentation_lora
    #The full names of the modules are of the form model.layers.11.mlp.gate_proj
    #And it is said about the argument target_modules :
    # When passing a list of strings, either an exact match will be performed
    # or it is checked if the name of the module ends with any of the passed strings.

    include_up_proj_str = "with_up_proj" if include_up_proj else "without_up_proj"

    lora_target_modules = []
    for i in num_layers:
        lora_target_modules.append(f"{i}.mlp.down_proj")
        if include_up_proj:
            lora_target_modules.append(f"{i}.mlp.up_proj")
        if include_gate_proj:
            lora_target_modules.append(f"{i}.mlp.gate_proj")

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules
    )

    # 3. Training arguments

    weight_decay = 2e-4 # 2e-4 ACCORDING TO DAMA
    max_grad_norm = 5e-5 # 5e-5 ACCORDING TO DAMA

    #optim = "paged_adamw_32bit
    warmup_ratio = 0.05
    #lr_scheduler_type = "cosine"
    #warmup_steps, max_steps

    #group_by_length = True

    #evaluation_strategy = "epoch"
    #save_strategy = "epoch"
    logging_steps = 1
    #fp16 = True
    #save_safetensors = True
    #seed = 42

    OUTPUT_DIR = f"../finetuned_models/{finetuned_model_name}"

    training_arguments = TrainingArguments(
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        #optim=optim,
        warmup_ratio=warmup_ratio,
        #lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #group_by_length=group_by_length,
        #evaluation_strategy=evaluation_strategy,
        #save_strategy=save_strategy,
        logging_steps=logging_steps,
        #save_safetensors=save_safetensors,
        #seed=seed,
        output_dir=OUTPUT_DIR
    )

    #Test dataset
    test_dataset = load_dataset('json',
                                data_files=f'../created_datasets/finetuning_datasets/{test_dataset_name}.jsonl',
                                split='train')
    #We are supposed to tokenize it
    test_dataset = test_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

    train_dataset = load_dataset('json',
                                 data_files=f'../created_datasets/finetuning_datasets/'
                                            f'{train_dataset_name}.jsonl',
                                 split='train')
    #We are supposed to tokenize it
    train_dataset = train_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

    # 2. We perform finetuning

    # a. We create a new Peft model, because Peft models are directly modified during training
    model = get_peft_model(base_model, peft_config)
    if verbose:
        print("Obtained Peft model")
        model.print_trainable_parameters()

    # b. The actual training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    if verbose:
        print("Beginning training")

    start_training = time.time()
    trainer.train()
    end_training = time.time()

    if verbose:
        print("Successful training")
        print(f"{end_training - start_training} s for {num_epochs} train epochs")

    # 4. Push to hub
    huggingface_hub.login(WRITING_TOKEN) #writing token
    if verbose:
        print("Self login successful")

    # 4.a. Pushing Peft model
    model.push_to_hub(finetuned_model_name,
                      revision=revision,
                      commit_description=revision)
    if verbose:
        print("Pushed Peft model successfully")


def push_merged_model_to_hub(base_model_name, finetuned_model_name, revision, verbose=True):
    model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                 torch_dtype=torch.float16,
                                                 # low_cpu_mem_usage=True,
                                                 # device_map='auto'
                                                 )
    model = PeftModel.from_pretrained(model, f"PaulM2000/{finetuned_model_name}", revision=revision)
    model = model.merge_and_unload()
    if verbose:
        print("Local merging successful")

    huggingface_hub.login(WRITING_TOKEN)  # writing token
    model.push_to_hub(f"merged_{finetuned_model_name}",
                      revision=revision,
                      commit_description=revision)
    if verbose:
        print(f"Pushing of the merged model {revision} successful")
