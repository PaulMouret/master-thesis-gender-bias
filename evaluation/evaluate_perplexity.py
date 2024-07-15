import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.inference_utils import get_perpexity_from_names
from utils.global_utils import store_obj_in_jsonfile
from utils.constants_utils import *


model_names = [BASE_MODEL_NAME,
               "dama_model_18-22_Llama-2-7b-hf_stereotyped_related"] + \
              [f"PaulM2000/peft_model_18-22_Llama-2-7b-hf {train_data}_{str(lr)}_{20}"
               for lr in (1e-5, 1e-6) for train_data in ("stereotyped_related", "stereotyped")]

for m in model_names:
    print(f"\n####################################\n{m}\n")
    ppl = get_perpexity_from_names(m)
    print(f"\nperplexity : {ppl}\n")

    filename = f"{m.split('/')[-1]}_ppl"
    dict_ppl = {m.split('/')[-1]: ppl}
    store_obj_in_jsonfile(dict_ppl, '../saved_objects/evaluation/perplexities/', filename)
