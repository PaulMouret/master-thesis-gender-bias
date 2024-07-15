import numpy as np
from matplotlib import pyplot as plt
from datasets import load_dataset

import sys

from utils.global_utils import bias_score, bias_score_from_couple
from plotting_utils import heatmap, annotate_heatmap


# ! BE CAREFUL : compared to my code, for DAMA code (whose causal tracing results come from)
# the index for pronouns is not the same : it is  for 0 she and 1 for he
#If needed, I can check DAMA notebook averaged_casual_tracing_severing_mlps

known_datasets = ["stereotyped_200000"]
dataset_size = 1000
each_sentence_bool = True
#Previous variables : known_datasets = ["stereotyped", "stereotyped_related"] ; dataset_size = 100

for knwon_dataset in known_datasets:
    res_directory = f"{knwon_dataset}_{dataset_size}/"
    print(f"\n{res_directory}")
    # 0. We load the data
    data = load_dataset('json', data_files=f'../created_datasets/causal_tracing_datasets/'
                                           f'results_{knwon_dataset}_{dataset_size}.jsonl',
                        split='train')
    print(f"{len(data)} elements before filtering")
    #To avoid extreme results in stereotypes
    data = data.filter(lambda x: np.abs(x["bias_score"] -
                                        bias_score(x["null"]["low_score"][1], x["null"]["low_score"][0])) >= 0.01)
    print(f"{len(data)} elements after filtering")

    #data = data.filter(lambda x, i: i ==1, with_indices=True)

    #data contains the keys :
    # prompt, subject, subject_index, subject_start_end, subject_male_stereotyped,
    # dialogue_context, prob_he, prob_she, relevance_score, bias_score, male_stereotype
    # null : {scores: (nb_tokens, 32, 3), low_score: (3,), high_score: (3,)}
    # mlp : {scores: (nb_tokens, 32, 3), low_score: (3,), high_score: (3,)}
    # attn : {scores: (nb_tokens, 32, 3), low_score: (3,), high_score: (3,)}

    # Corrupted biases (they are independent of layers and tokens),
    # and independent of the type of layer, so we can take them from "null" for instance
    corrupted_probs = np.array([context["null"]["low_score"] for context in data]) #shape (nb_contexts, 3)
    corrupted_biases = bias_score(corrupted_probs[:, 1], corrupted_probs[:, 0]) #shape (nb_contexts,)
    biases = np.array(data["bias_score"]) #shape (nb_contexts,)
    stereotypical_directions = biases - corrupted_biases

    # 1. Global plots

    for kind in ["null", "mlp", "attn"]:
        #kind_str
        if kind=="mlp":
            kind_str = "mlp layer"
        elif kind == "attn":
            kind_str = "attention layer"
        else:
            kind_str = "layer"
        title = f"{kind_str}"
        print(f"\n##### {title.upper()} #####")

        ls_avg_diff_biases = [] # ls stands for last subject token
        lt_avg_diff_biases = [] # lt stands for last token

        ls_avg_abs_diff_biases = []
        lt_avg_abs_diff_biases = []

        ls_avg_diff_stereotypes = []
        lt_avg_diff_stereotypes = []

        ls_avg_diff_he = []
        ls_avg_diff_she = []
        lt_avg_diff_he = []
        lt_avg_diff_she = []
        for layer in range(32):
            ls_restored_biases = [] #the vector of the restored biases of the last subject token for this layer
            lt_restored_biases = []  # the vector of the restored biases of the last token for this layer

            ls_restored_he = []
            ls_restored_she = []
            lt_restored_she = []
            lt_restored_he = []
            for id_example, example in enumerate(data):
                bias_example = biases[id_example]
                corrupted_bias_example = corrupted_biases[id_example]

                beginning, end = example["subject_start_end"]
                end -= 1
                ls_restored_probs = example[kind]['scores'][end][layer]
                lt_restored_probs = example[kind]['scores'][-1][layer]

                ls_restored_he.append(ls_restored_probs[1])
                ls_restored_she.append(ls_restored_probs[0])
                lt_restored_he.append(lt_restored_probs[1])
                lt_restored_she.append(lt_restored_probs[0])

                ls_restored_biases.append(bias_score(ls_restored_probs[1], ls_restored_probs[0]))
                lt_restored_biases.append(bias_score(lt_restored_probs[1], lt_restored_probs[0]))

            ls_diff_biases = np.array(ls_restored_biases) - corrupted_biases
            lt_diff_biases = np.array(lt_restored_biases) - corrupted_biases

            ls_diff_stereotypes = ls_diff_biases * (1/stereotypical_directions)
            lt_diff_stereotypes = lt_diff_biases * (1/stereotypical_directions)

            #For debugging
            weird_indices = np.arange(len(biases))[np.abs(lt_diff_stereotypes) > 20]
            for weird_i in weird_indices:
                print(f"layer {layer} lt_diff_stereotype is {lt_diff_stereotypes[weird_i]} for"
                      f" context {weird_i} :\n{data['prompt'][weird_i]} "
                      f"(stereotype {data['subject'][weird_i]}) (stereotypical direction {stereotypical_directions[weird_i]})")

            ls_avg_diff_stereotypes.append(np.mean(ls_diff_stereotypes))
            lt_avg_diff_stereotypes.append(np.mean(lt_diff_stereotypes))

            ls_avg_diff_biases.append(np.mean(ls_diff_biases))
            lt_avg_diff_biases.append(np.mean(lt_diff_biases))

            lt_avg_abs_diff_biases.append(np.mean(np.abs(lt_diff_biases)))
            ls_avg_abs_diff_biases.append(np.mean(np.abs(ls_diff_biases)))

            ls_avg_diff_he.append(np.mean(np.array(ls_restored_he) - corrupted_probs[:, 1]))
            ls_avg_diff_she.append(np.mean(np.array(ls_restored_she) - corrupted_probs[:, 0]))
            lt_avg_diff_he.append(np.mean(np.array(lt_restored_he) - corrupted_probs[:, 1]))
            lt_avg_diff_she.append(np.mean(np.array(lt_restored_she) - corrupted_probs[:, 0]))

        # Now all *_avg_* vectors have length (32,)

        #Now we have everything, we create the objects to plot
        y_labels = ["last stereotyped word token", "last context token"]
        x_labels = np.arange(32)

        # a. Bias (coloured) heatmap
        print(f"Bias (coloured) heatmap {kind_str}")
        print(np.around(ls_avg_diff_biases, 2))
        print(np.around(lt_avg_diff_biases, 2))
        plot_data = np.array([ls_avg_diff_biases,
                              lt_avg_diff_biases])
        #And we proceed to plotting based on plotting_utils
        fig, ax = plt.subplots()
        im, cbar = heatmap(plot_data, y_labels, x_labels, title, ax=ax,
                           cmap='RdBu',
                           vmin=(-0.6), vmax=0.6,
                           cbarlabel="indirect bias effect") #gray for grayscale
        #texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        plt.savefig(f"../saved_objects/causal_tracing/{res_directory}indirect_bias_effect_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # b. Absolute bias (grayscale) heatmap
        print(f"Absolute bias (grayscale) heatmap {kind_str}")
        print(np.around(ls_avg_abs_diff_biases, 2))
        print(np.around(lt_avg_abs_diff_biases, 2))
        abs_plot_data = np.array([ls_avg_abs_diff_biases,
                                  lt_avg_abs_diff_biases])
        fig, ax = plt.subplots()
        im, cbar = heatmap(abs_plot_data, y_labels, x_labels, title, ax=ax,
                           cmap='binary',
                           vmin=0., vmax=1.,
                           cbarlabel="absolute indirect bias effect")  # binary
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        plt.savefig(f"../saved_objects/causal_tracing/{res_directory}abs_indirect_bias_effect_{kind}.png", dpi=200,
                    bbox_inches="tight")
        plt.close()

        # c. Stereotype (coloured) heatmap
        print(f"Stereotype (coloured) heatmap {kind_str}")
        print(np.around(ls_avg_diff_stereotypes, 2))
        print(np.around(lt_avg_diff_stereotypes, 2))
        plot_data = np.array([ls_avg_diff_stereotypes,
                              lt_avg_diff_stereotypes])
        # And we proceed to plotting based on plotting_utils
        fig, ax = plt.subplots()
        im, cbar = heatmap(plot_data, y_labels, x_labels, title, ax=ax,
                           cmap='RdBu',
                           vmin=(-1.), vmax=1., #it takes value till -11., but those are exceptions
                           cbarlabel="indirect prostereotypical effect")  # gray for grayscale
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        plt.savefig(f"../saved_objects/causal_tracing/{res_directory}indirect_stereotype_effect_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # d. Pronoun heatmaps
        print(f"Pronoun heatmaps {kind_str}")
        print(np.around(ls_avg_diff_he, 2))
        print(np.around(lt_avg_diff_he, 2))

        # d1. he
        he_plot_data = np.array([ls_avg_diff_he,
                                 lt_avg_diff_he])
        fig, ax = plt.subplots()
        im, cbar = heatmap(he_plot_data, y_labels, x_labels, title, ax=ax,
                           cmap='binary',
                           vmin=0., vmax=0.06,
                           cbarlabel="indirect 'he' effect")
        # ! the scale is not the same as for she
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        plt.savefig(f"../saved_objects/causal_tracing/{res_directory}indirect_he_effect_{kind}.png", dpi=200,
                    bbox_inches="tight")
        plt.close()

        # d1. she
        print(np.around(ls_avg_diff_she, 2))
        print(np.around(lt_avg_diff_she, 2))
        she_plot_data = np.array([ls_avg_diff_she,
                                 lt_avg_diff_she])
        fig, ax = plt.subplots()
        im, cbar = heatmap(she_plot_data, y_labels, x_labels, title, ax=ax,
                           cmap='binary',
                           vmin=0., vmax=0.06,
                           cbarlabel="indirect 'she' effect")
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        plt.savefig(f"../saved_objects/causal_tracing/{res_directory}indirect_she_effect_{kind}.png", dpi=200,
                    bbox_inches="tight")
        plt.close()

    # 2. For each sentence
    if each_sentence_bool:
        print("For each sentence")
        for id_example, example in enumerate(data):
            if id_example % 10 == 0:
                bias_example = example["bias_score"]
                corrupted_bias_example = bias_score(example['null']['low_score'][1], example['null']['low_score'][0])
                #Debugging
                #print(f"bias score {bias_example} ;
                # calculated {bias_score(example['null']['high_score'][1], example['null']['high_score'][0])}")
                for kind in ["null", "mlp", "attn"]:
                    # kind_str
                    if kind == "mlp":
                        kind_str = "mlp layer"
                    elif kind == "attn":
                        kind_str = "attention layer"
                    else:
                        kind_str = "layer"
                    title = f"{example['prompt']} ({example['subject']})\n{kind_str}"

                    ls_restored_biases = []
                    lt_restored_biases = []

                    for layer in range(32):
                        beginning, end = example["subject_start_end"]
                        end -= 1
                        ls_restored_probs = example[kind]['scores'][end][layer]
                        lt_restored_probs = example[kind]['scores'][-1][layer]

                        ls_restored_biases.append(bias_score(ls_restored_probs[1], ls_restored_probs[0]))
                        lt_restored_biases.append(bias_score(lt_restored_probs[1], lt_restored_probs[0]))

                    #Plot
                    x_axis = np.arange(32)

                    plt.plot(x_axis, ls_restored_biases, '--bo', markersize=4,
                             label='last subject token restored bias')
                    plt.plot(x_axis, lt_restored_biases, '--go', markersize=4,
                             label='last context token restored bias')
                    plt.hlines(bias_example, xmin=0, xmax=31, colors='r', label='bias score')
                    plt.hlines(corrupted_bias_example, xmin=0, xmax=31, colors='black', label='corrupted bias score')
                    plt.xlabel(kind_str)
                    plt.ylabel("Bias")
                    plt.title(title)
                    plt.legend()

                    plt.savefig(f"../saved_objects/causal_tracing/{res_directory}for_each_sentence/indirect_bias_effect_{kind}_{id_example}.png",
                                dpi=200,
                                bbox_inches="tight")
                    plt.close()
