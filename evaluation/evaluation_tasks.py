from prettytable import PrettyTable
import os

from utils.global_utils import load_obj_from_jsonfile, prettytable_to_latex
from utils.constants_utils import *


# First, we compute beforehand the names of all models and their paths
path_results = "../results_05_20/"

paths = ["original/llama2_7B/"]
names = ["LLaMA 2 7B"]
start_layer = 18
learning_rate = 1e-5
for num_layers in [5, 12]: #[5, 6, 8, 10, 12, 14]
    for method in ["DAMA", "LoRA"]:
        for train_data in ["stereotyped_related", "stereotyped", "professions"]:
            if method == "DAMA":
                train_data_dama = train_data
                if train_data == "stereotyped_related":
                    train_data_dama = "stereotyped-related"
                elif train_data == "professions":
                    train_data_dama = "tomasz-professions"
                path = f"{train_data_dama}/DAMA_L/llama2_7B_{num_layers}L/"
                name = f"{num_layers}L / DAMA / {train_data}"
                paths.append(path)
                names.append(name)
            elif method == "LoRA":
                for str_gate_proj in ['_with_gp']: #, ''
                    peft_model_name = f"peft_model_{start_layer}-{start_layer + num_layers - 1}{str_gate_proj}" \
                                      f"_{BASE_MODEL_NAME.split('/')[-1]} {train_data}_{str(learning_rate)}_{20}"
                    model_name_revision = peft_model_name.split()
                    model_name, revision = model_name_revision[0], model_name_revision[1]

                    path = f"PEFT/{model_name}/{revision}/"
                    name = f"{num_layers}L / LoRA / {train_data}" #{str_gate_proj}
                    paths.append(path)
                    names.append(name)
            else:
                print(f"Unknown method : {method}")


# The list of tasks (without the extension of their result file)
list_tasks = ["res_causal_lm_wikitext_wikitext-103-raw-v1",
              "res_coref_anti_type1_test",
              "res_coref_anti_type2_test",
              "res_coref_pro_type1_test",
              "res_coref_pro_type2_test",
              "res_gen_test_dama",
              "res_qa_ARC-Challenge-Test",
              "res_qa_ARC-Easy-Test",
              "res_stereoset_stereoset_dev"]

list_fields = [
    ["mean_perplexity"],
    ["m_acc", "f_acc", "total_acc"],
    ["m_acc", "f_acc", "total_acc"],
    ["m_acc", "f_acc", "total_acc"],
    ["m_acc", "f_acc", "total_acc"],
    ["joint_slope_s", "joint_slope_f", "joint_intercept", "joint_r2"],
    ["per_token_prob_root", "per_token_prob", "per_char_prob", "normed_prob", "unnormed_prob"],
    ["per_token_prob_root", "per_token_prob", "per_char_prob", "normed_prob", "unnormed_prob"],
    ["gender_LM", "gender_SS", "gender_ICAT"]
]

for task, fields in zip(list_tasks, list_fields):
    print(f"\n### {task}")
    #We have to create the table ; for this we need the list of column names
    some_res = load_obj_from_jsonfile(path_results + paths[0], task)
    #print(f"some_res : {some_res}")
    key_names = list(some_res.keys())
    #print(f"key_names : {key_names}")
    columns_names = ["Model"] + key_names
    #print(f"columns_names : {columns_names}")
    table = PrettyTable(columns_names)
    #print(f"empty table :\n{table}")

    #Now we can fill-in the table
    for path, name in zip(paths, names):
        model_path = path_results + path
        global_model_name = name

        if os.path.exists(model_path):
            res = load_obj_from_jsonfile(model_path, task)
            row = [global_model_name] + [round(res[k], 3) for k in key_names]
            #print(f"row : {row}")
            table.add_row(row)

    #The table is filled

    #Now we can save it
    chain = table.get_string(fields=["Model"] + fields)
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/evaluation/evaluation_tasks"
    with open(f"{repertory}/{task}_table.txt", 'w') as f:
        f.write(chain)

    #Latex equivalent
    prettytable_to_latex(f"{repertory}/{task}_table", first_column_bold=True)


#Now we make the table for WinoBias (coref)
#It is different because not all values are in the table : we have to compute them

for id_type in (1,2):
    print(f"\n### WinoBias type {id_type}")

    pro_task = f"res_coref_pro_type{id_type}_test"
    anti_task = f"res_coref_anti_type{id_type}_test"

    columns_names = ["Model", "accuracy", "delta_S", "delta_G"]
    table = PrettyTable(columns_names)

    #Now we can fill-in the table
    for path, name in zip(paths, names):
        model_path = path_results + path
        global_model_name = name

        if os.path.exists(model_path):
            pro_res = load_obj_from_jsonfile(model_path, pro_task)
            anti_res = load_obj_from_jsonfile(model_path, anti_task)

            row = [global_model_name,
                   round((pro_res["total_acc"] + anti_res["total_acc"])/2, 3),
                   round(pro_res["total_acc"] - anti_res["total_acc"], 3),
                   round((pro_res["m_acc"] + anti_res["m_acc"] - pro_res["f_acc"] - anti_res["f_acc"])/2, 3)]
            table.add_row(row)

    #The table is filled

    #Now we can save it
    chain = table.get_string()
    # I can use the fields argument of get_string to easily keep only columns of interest
    print(chain)

    repertory = "../saved_objects/evaluation/evaluation_tasks"
    with open(f"{repertory}/winobias_type_{id_type}_table.txt", 'w') as f:
        f.write(chain)

    prettytable_to_latex(f"{repertory}/winobias_type_{id_type}_table", first_column_bold=True)

#We make the global tables

#Global gender bias evaluation
columns_names = ["Model", "a_s", "a_f", "b", "sub_acc", "sub_delta_S", "sub_delta_G", "obj_acc", "obj_delta_S",
                 "obj_delta_G", "ss", "icat"]
table = PrettyTable(columns_names)
#Now we can fill-in the table
for path, name in zip(paths, names):
    model_path = path_results + path
    global_model_name = name

    if os.path.exists(model_path):

        row = [global_model_name]

        limi_res = load_obj_from_jsonfile(model_path, "res_gen_test_dama")
        limi_keys = ["joint_slope_s", "joint_slope_f", "joint_intercept"]
        limi_row = [round(limi_res[k], 3) for k in limi_keys]
        row += limi_row

        for id_type in (1, 2):
            pro_task = f"res_coref_pro_type{id_type}_test"
            anti_task = f"res_coref_anti_type{id_type}_test"

            pro_res = load_obj_from_jsonfile(model_path, pro_task)
            anti_res = load_obj_from_jsonfile(model_path, anti_task)

            wino_bias_row = [round(100 * (pro_res["total_acc"] + anti_res["total_acc"]) / 2, 1),
                             round(100 * (pro_res["total_acc"] - anti_res["total_acc"]), 1),
                             round(100 * (pro_res["m_acc"] + anti_res["m_acc"] - pro_res["f_acc"] - anti_res["f_acc"]) / 2, 1)]
            row += wino_bias_row

        stereoset_res = load_obj_from_jsonfile(model_path, "res_stereoset_stereoset_dev")
        stereoset_keys = ["gender_SS", "gender_ICAT"]
        stereoset_row = [round(stereoset_res[k], 1) for k in stereoset_keys]
        row += stereoset_row

        table.add_row(row)

# The table is filled
# Now we can save it
chain = table.get_string()
# I can use the fields argument of get_string to easily keep only columns of interest
print(chain)
repertory = "../saved_objects/evaluation/evaluation_tasks"
with open(f"{repertory}/global_bias_table.txt", 'w') as f:
    f.write(chain)
prettytable_to_latex(f"{repertory}/global_bias_table", first_column_bold=True)


#Global llm evaluation
columns_names = ["Model", "PPL", "ARC-E", "ARC-C"]
table = PrettyTable(columns_names)
#Now we can fill-in the table
for path, name in zip(paths, names):
    model_path = path_results + path
    global_model_name = name

    if os.path.exists(model_path):

        row = [global_model_name]

        perplexity_res = load_obj_from_jsonfile(model_path, "res_causal_lm_wikitext_wikitext-103-raw-v1")
        perplexity_keys = ["mean_perplexity"]
        perplexity_row = [round(perplexity_res[k], 1) for k in perplexity_keys]
        row += perplexity_row

        for arc_task in ["res_qa_ARC-Easy-Test", "res_qa_ARC-Challenge-Test"]:
            arc_res = load_obj_from_jsonfile(model_path, arc_task)
            arc_keys = ["normed_prob"]
            arc_row = [round(100 * arc_res[k], 1) for k in arc_keys]
            row += arc_row

        table.add_row(row)

# The table is filled
# Now we can save it
chain = table.get_string()
# I can use the fields argument of get_string to easily keep only columns of interest
print(chain)
repertory = "../saved_objects/evaluation/evaluation_tasks"
with open(f"{repertory}/global_llm_table.txt", 'w') as f:
    f.write(chain)
prettytable_to_latex(f"{repertory}/global_llm_table", first_column_bold=True)
