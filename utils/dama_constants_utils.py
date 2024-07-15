#For all files
DAMA_BASE_MODEL_NAME = "huggyllama/llama-7b" #"meta-llama/Llama-2-7b-hf" ; "huggyllama/llama-7b" is the original

#The names of the parameters for which DAMA finetuning datasets have been created so far
DAMA_STRATEGIES = ["random", "neutral", "opposite"]
DAMA_RANDOM_SEEDS = [42]

#The latest committed versions of Peft models (with DAMA dataset)
#Note that a Peft model and its corresponding merged model have the same revision name
#Be careful : when loading a (Peft or merged) model, it is this latest version that is automatically loaded
DAMA_PEFT_REVISION = "v6" #v4 was the latest for LLama 2 (1e-6) ; v4 is the latest for LLama (1e-4)
DAMA_PEFT_COMMIT = "With lr=5e-5"