#For all files
WRITING_TOKEN = '' # HF writing token : should be changed manually (and kept secret)
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-2-7b-hf" ; "huggyllama/llama-7b" is the original
RELEVANCE_THRESHOLD = 0.0349932518 #see benchmark.py
OCCURRENCE_THRESHOLD = 10 #see analysis_distribution_bias.py
FEMALE_THRESHOLD = -0.437 #see analysis_stereotypical_scores.py
MALE_THRESHOLD = 0.392 #see analysis_stereotypical_scores.py
STEREOTYPICAL_SHIFT = -0.1089
FINAL_STD = 0.016960975157652373 #0.01334381103515625 #see test_hypothesis_emb_variance_2.py
FINAL_FACTOR = 2.5 #see test_optimal_alpha.py

#The names of the parameters for which inference (adding_probs_to_subsets.py) has been done so far
#As we first check existence of inferred datasets before performing datasets,
#it is fine to let old parameters in list : they will not cause additional computations
#Note that at the end I also used it with other files
CONTEXT_DATASET_NAME = "contexts_bookcorpus"
DATASET_RANDOM_SEED = 42
BATCH_SIZE = 16 #see get_max_batch_size.py
SUBSET_SIZES = [1000, 50000, 200000]
#should be in increasing order, because some programs call SUBSET_SIZES[0] or SUBSET_SIZES[-1]


#For finetuning

#The names of the parameters for which finetuning datasets have been created so far
FINETUNING_STRATEGIES = ["random", "neutral", "opposite"] #["random", "neutral", "opposite"]
CHOICES_STEREOTYPICAL = [True, False] #[True, False]
FINETUNING_RANDOM_SEED = 42 #for random strategy

# 1. All linear all datasets

AL_AD_FT_MODEL_NAME = f"peft_model_all_linear_all_datasets_{BASE_MODEL_NAME.split('/')[-1]}"
HUB_AL_AD_FT_MODEL_NAME = f"PaulM2000/{AL_AD_FT_MODEL_NAME}"
#Hopefully we won't need to merge models
#merged_finetuned_model_name = f"merged_{finetuned_model_name}"
#hub_merged_finetuned_model_name = f"PaulM2000/{merged_finetuned_model_name}"
AL_AD_FT_REVISIONS = []
AL_AD_LEARNING_RATE = 1e-6
AL_AD_NUM_EPOCHS = 5
for choice_stereotypical in CHOICES_STEREOTYPICAL:
    for strategy in FINETUNING_STRATEGIES:
        if choice_stereotypical:
            stereotype_str = "stereotyped"
        else:
            stereotype_str = "nonstereotyped"
        full_strategy = f'{stereotype_str}_{strategy}'
        revision = f"{full_strategy}_{str(AL_AD_LEARNING_RATE)}_{AL_AD_NUM_EPOCHS}"
        AL_AD_FT_REVISIONS.append(revision)
AL_AD_FT_MODEL_NAMES = [HUB_AL_AD_FT_MODEL_NAME + " " + rev for rev in AL_AD_FT_REVISIONS]

# 2. All linear
ALL_LINEAR_MODEL_NAME = f"peft_model_all_linear_{BASE_MODEL_NAME.split('/')[-1]}"
HUB_ALL_LINEAR_MODEL_NAME = f"PaulM2000/{ALL_LINEAR_MODEL_NAME}"

ALL_LINEAR_STRATEGY = "random"
ALL_LINEAR_STEREOTYPICAL = True
ALL_LINEAR_LEARNING_RATE = 1e-6
ALL_LINEAR_NUM_EPOCHS = 20
ALL_LINEAR_FULL_STRATEGY = f'{"stereotyped" if ALL_LINEAR_STEREOTYPICAL else "nonstereotyped"}_{ALL_LINEAR_STRATEGY}'
ALL_LINEAR_REVISION = f"{ALL_LINEAR_FULL_STRATEGY}_{str(ALL_LINEAR_LEARNING_RATE)}_{ALL_LINEAR_NUM_EPOCHS}"

ALL_LINEAR_MODEL_NAMES = [HUB_ALL_LINEAR_MODEL_NAME + " " + ALL_LINEAR_REVISION]

# 3. Limi layers
LIMI_MODEL_NAME = f"peft_model_limi_layers_{BASE_MODEL_NAME.split('/')[-1]}"
HUB_LIMI_MODEL_NAME = f"PaulM2000/{LIMI_MODEL_NAME}"

LIMI_STRATEGY = "random"
LIMI_STEREOTYPICAL = True
LIMI_LEARNING_RATE = 1e-6
LIMI_NUM_EPOCHS = 20
LIMI_FULL_STRATEGY = f'{"stereotyped" if LIMI_STEREOTYPICAL else "nonstereotyped"}_{LIMI_STRATEGY}'
LIMI_REVISION = f"{LIMI_FULL_STRATEGY}_{str(LIMI_LEARNING_RATE)}_{LIMI_NUM_EPOCHS}"

LIMI_MODEL_NAMES = [f"PaulM2000/merged_{LIMI_MODEL_NAME}" + " " + LIMI_REVISION]

#################################################

#The models to be used for inference (full_inference/adding_probs_to_subset.py)
MODEL_NAMES = [BASE_MODEL_NAME]

#The models to be used for testing finetuning (full_inference/test_finetuning.py)
TEST_FINETUNING_MODEL_NAMES = [BASE_MODEL_NAME] + ALL_LINEAR_MODEL_NAMES + LIMI_MODEL_NAMES #+ AL_AD_FT_MODEL_NAMES
