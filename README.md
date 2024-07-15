As there are many files, and as I may have to run some parts of the code several times, with differents datasets,
here is the order in which I ran these files.
Their status (typically the parameters they have been run with) is written **in bold**.
Important remarks are written *in italic*.
Between parentheses are function modules (which are not run directly).

As .jsonl files usually correspond to large files and thus are discarded by .gitignore, the .jsonl results 
should be imported manually. The tag *(JSONL)* appears when one is created.

The scripts files beginning with all_* regroup several programs, to be more efficient during running.

empty.txt files are here to force git to add empty repositories

# Getting source files

* save_wikitext.py

* I downloaded other files manually from various places ; 
  and processed BookCorpus into a sentence list .txt file using make_sentlines.py

# Utils

* (global_utils.py)

* (inference_utils.py / dama_based_inference_utils.py) : called during inference, in particular adding_probs_to_subset.py

* (constants_utils.py / dama_constants_utils.py)

* (noising_utils.py) : called during noising experiment

* (finetuning_utils.py) : functions for finetuning, pushing merged model to the hub 
  and printing model parameters

* (nethook.py) : called during noising experiment

* (perplexity.py) : called in inference_utils.py, to compute perplexity of a model

# Context dataset creation

* (dataset_creation_utils.py)

* gender_of_persons.py / firstnames.py / manual_gendered_words.py / stereotyped_occupations.py : 
  to get corresponding utils datasets and statistics
  *(note that manual_gendered_words.py can be progressively updated and rerun, so everytime the whole process
  has to be done again ;
  moreover there are two types of gendered firstnames : English / English or of length > 3)*
  **Up-to-date (ie. manual_gendered_words.py has been run with latest changes)**

* creating_dataset.py / creating_other_datasets.py : 
  to create the context datasets and the corresponding statistics
  *(JSONL)*
  **Up-to-date
  (ie. taking latest changes of utils_datasets into account)**
  
* (statistics_dataset_creation_utils.py)
  
* statistics_visualization.py : 
  to print and save as .png files the statistics corresponding to the created datasets
  **Up-to-date (ie. performed on the latest context dataset that has been created)**
  
* update_context_dataset_with_manual.py : 
  to remove contexts that contain a word specified in manual_gendered : it avoids 
  reconstructing the whole dataset, when we just remove contexts and just a few of them)
  *However, note that, this way, statistics about the number of contexts are not the same
  anymore ; for this the whole dataset creation should be rerun*
  *(JSONL)*
  
# Preliminary work

## Batch size analysis

* get_max_batch_size.py : 
  to get the optimal batch_size that should be used with the model
  **The optimal batch_size are, in decreasing order, 64, 32, 16. In particular, 16 is better than 24
  (maybe because it is a power of 2) and the model (on AIC GPU) can manage at least batch size 64.
  However in some cases 64 is too much for memory.
  NOT UP-TO-DATE with latest dataset (but should get similar results)**

## Deterministic analysis

* is_deterministic.py : 
  to make sure the Llama 2 model is deterministic
  **It is indeed.
  NOT UP-TO-DATE with latest dataset (but should get similar results)**
  
## Padding analysis

* (padding_analysis_utils.py)

* padding_analysis.py : 
  to get the inference results for padded variants (saved in .json files)
  and the histograms of corresponding statistics (saved as .png files).
  **NOT UP-TO-DATE with latest dataset (but should get similar results)**
  
* padding_side_analysis.py : 
  to compare between padding on the left (the one I chose) and on the right
  
# Evaluating bias

* full_inference.py : 
  to get the pronoun probabilities (stored in a .json file) for a whole dataset.
  *Except the inference time, that may be useful in a first step to compare parameters,
  it is simply less informative than adding_probs_to_subsets.py, so at some point it should
  not be used anymore.*
  
* adding_probs_to_subsets.py : 
  to create subsets with inference results (probs, bias and relevance scores),
  in order not to do it everytime
  *(JSONL)*
  **Run for LLaMA 2 7B, subset_sizes 1000, 50,000 and 200,000**
  
* (benchmark_utils.py)
  
* benchmark.py / complementary_benchmark_code.py : 
  to get statistics (about probabilities, bias and relevance) and plots (saved as .png files) 
  about a subset of created contexts. We use the results from naive_full_inference.
  **Up-to-date (ie. performed on all inference datasets that have been created)**

# Noising experiment

## Determining hyperparameters for noising experiment

* test_hypothesis_emb_variance.py : to find the value of sigma_t (final_std) 
  and store token embeddings
  
* test_hypothesis_emb_variance_2.py : to proceed to statistical testing on stored token embeddings

* find_optimal_alpha.py : performs several noising experiments with various values for alpha

* test_optimal_alpha.py : determines the best alpha based on results of find_optimal_alpha.py

## Running the actual noising experiment

* final_noising_experiment.py : performs the noising experiment (with the optimal value for alpha)
**(JSONL)**

## Analyzing results of the noising experiment

* analysis_final_noising_experiment.py : analyses global results of final_noising_experiment.py ; 
  distribution of corrupted biases and relevances, distribution of biases and relevance differences,
  analysis of crucial words

* analysis_distribution_bias_diff.py : analysis the distribution of bias difference, in order to 
  determine the reliability conditions for the stereotypical score

* analysis_stereotypical_scores.py : analyses stereotypical scores

* looking_for_contexts.py : a util file enabling to filter contexts
  respecting various conditions, and print them out
  
# Related words

* create_dataset_to_parse.py : creates the .txt file that will be parsed *(gitignored !)*

* parsing_script.sh : performs parsing and generates an output file *(gitognored !)*

* format_parsed_contexts.py : includes results in the context dataset that has been parsed *(JSONL)*

* (find_related_words_utils.py)

* find_related_words.py : to add the list of words related to the pronoun for each context 
  of a given dataset.
  
* check_related_words.py : prints contexts and their related words

# Causal tracing

* create_known_datasets.py *(gitignored !)*

* (causal_trace.py / causal_tracing_globals.py /causal_tracing_utils.py / gender_trace.py
  knowns.py) *(because this code is copied, with slight modifications, 
  from Tomasz Limisiewicz DAMA code, the conventions may be different from the rest of the code,
  and thus it should not interfere with the rest of the code)*
  
* trace.py *(several scripts, depending on the dataset used to perform causal tracing)*

* (plotting_utils.py)

* plotting.py

# DAMA Finetuning

* create_datasets_for_dama_finetuning.py : creates datasets for finetuning 
  based on "DAMA dataset", according to various parameters (in particular strategies)
  
* dama_peft_finetuning.py : performs finetuning based on a DAMA finetuning dataset 
  and pushes the corresponding Peft model to HF
  
* pushing_dama_merged_model_to_hub.py : creates the merged model corresponding to 
  previous Peft model and pushes it to the hub
  *The files full_inference.py and print_parameters_of_models.py should be run 
  only after the merged model corresponding to current DAMA_PEFT_REVISION is pushed,
  else it will raise an error.*

# Finetuning

*Ideally, we would like to avoid using the merged model, because it costs additional
  time and memory (for pushing and loading)*

* create_datasets_for_finetuning.py : creates a dataset for finetuning
  (ie. with relevant sentences, by adding them "he" and "she" at the end :
  6 possible strategies are used)
  
* peft_finetuning_all_linear_all_datasets.py :
  performs finetuning on all linear layers, one finetuning for each finetuning dataset,
  and pushes the corresponding Peft model to HF
  
* peft_finetuning_all_linear.py : performs finetuning on all linear layers 
  and pushes the corresponding Peft model to HF
  
* peft_finetuning_limi_layers.py : performs finetuning on linear layers highlighted
  by Tomasz Limisiewicz in DAMA paper, and pushes the corresponding Peft model to HF
  
# Evaluation

* evaluate_bias_relevance.py : generates inference on test finetuning dataset, for several models

* table_from_bias_relevance.py : generates the table of bias and relevance based on test data inference

* table_he_she.py : prints he/she prediction for some sentences based on test data inference
  
* evaluate_perplexity.py : evaluates the perplexity of several models
  
* evaluation_tasks.py : generates table from evaluation tasks results .json files
  
* evaluation_top_k.py : prints top k prediction for some sentences