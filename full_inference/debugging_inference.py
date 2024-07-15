import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.inference_utils import *
from utils.dama_based_inference_utils import *
from utils.constants_utils import *


#The code is very similar to adding_probs_to_subset.py

model_names = [BASE_MODEL_NAME,
               "dama_model_18-22_Llama-2-7b-hf_stereotyped_related",
               f"PaulM2000/peft_model_18-22_Llama-2-7b-hf stereotyped_related_{str(1e-5)}_{20}",
               ]
tokenizer = load_tokenizer(BASE_MODEL_NAME)

#For later
batch_size = BATCH_SIZE
finetuning_dataset_name = f"finetuning_datasets/test_ft"


for m in model_names:
    print(f"\n####################################\n{m}\n")

    strings = ["\"Why are you here, Milagro?\"",
               "Trapper noticed that Sole was swimming as fast as",
               "A tall blond had a crew clear the area, and then",
               "But that voice of disapproval was growing more distant as"]

    print(f"My inference")
    my_inf = get_pronouns_probabilities_from_strings(strings, model_name=m, padding=True,
                                                     get_predictions=False, verbose=True)
    print(f"Dama inference")
    model = load_model(m)
    dama_inf = dama_inference(model, tokenizer, strings)
