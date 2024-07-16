# Evaluation of gender bias of Large Language Models in natural contexts

A few variables, such as huggingface login tokens and GPU devices in .sh scripts,
should be modified manually for the reader to use them.
Moreover, for storage reasons, some source datasets
and other folders (such as results_05_20/) are not included in this
git repository and should be imported manually.
Note that, when storing results files, we often assume that the corresponding
folder already exists; hence, it should be created in advance, or the code should be updated.

Here is the organisation and order in which the code should be run.
Between parentheses are function modules (which are not run directly).
As .jsonl files usually correspond to large files and thus are discarded by .gitignore, the .jsonl results 
should be imported manually. The tag *(JSONL)* appears when one is created.

## Getting source files

* Source files should be imported manually into source_datasets/ : 
  train_dama.json from [DAMA](https://github.com/tomlimi/DAMA)
  and out_txts/ the folder containing .txt files that compose BookCorpus.
  Note that
  firstnames.csv from [firstname-database
](https://github.com/MatthiasWinkelmann/firstname-database),
  gender_of_persons.json from [gendered_words
](https://github.com/ecmonsen/gendered_words)
  and professions.json from [DAMA](https://github.com/tomlimi/DAMA)
  already are in the git repository.

* make_sentlines.py : to transform BookCorpus (saved in out_txts/) 
  into a sentence list .txt file
  
* save_wikitext.py : downloads the dataset used to compute perplexity

## DAMA utils

* (dama_l_hparams.py / model_utils.py) : for loading a DAMA-finetuned model

## Utils

* (global_utils.py)

* (inference_utils.py / dama_based_inference_utils.py) : called during inference, in particular adding_probs_to_subset.py

* (constants_utils.py / dama_constants_utils.py)

* (noising_utils.py) : called during noising experiment

* (finetuning_utils.py) : functions for finetuning, pushing merged model to the hub 
  and printing model parameters

* (nethook.py) : called during noising experiment

* (perplexity.py) : called in inference_utils.py, to compute perplexity of a model

## Context dataset creation

* (dataset_creation_utils.py)

* gender_of_persons.py / firstnames.py / manual_gendered_words.py / stereotyped_occupations.py : 
  to get corresponding utils datasets and statistics.
  Note that manual_gendered_words.py can be progressively updated and rerun.

* creating_dataset.py : 
  to create the context dataset and the corresponding statistics
  *(JSONL)*
  
* (statistics_dataset_creation_utils.py)
  
* statistics_visualization.py : 
  to print and save as .png files the statistics corresponding to the created datasets
  
* update_context_dataset_with_manual.py : 
  to remove contexts that contain a word specified in manual_gendered : it avoids 
  reconstructing the whole dataset, when we just remove contexts and just a few of them).
  However, note that, this way,
  statistics about the number of contexts are not updated ;
  for this the whole dataset creation should be rerun.
  *(JSONL)*
  
## Preliminary work (optional)

### Batch size analysis

* get_max_batch_size.py : 
  to get the optimal batch_size that should be used with the model

### Deterministic analysis

* is_deterministic.py : 
  to make sure the model is deterministic
  
### Padding analysis

* (padding_analysis_utils.py)

* padding_analysis.py : 
  to get the inference results for padded variants (saved in .json files)
  and the histograms of corresponding statistics (saved as .png files).
  
* padding_side_analysis.py : 
  to compare between padding on the left (the one I choose) and on the right
  
## Evaluating bias (full inference)
  
* adding_probs_to_subsets.py : 
  to create subsets with inference results (probs, bias and relevance scores),
  in order not to do it everytime
  *(JSONL)*
  
* (benchmark_utils.py)
  
* benchmark.py / complementary_benchmark_code.py : 
  to get statistics (about probabilities, bias and relevance) and plots (saved as .png files) 
  about a subset of created contexts.

## Stereotype analysis (noising experiment)

### Determining hyperparameters for noising experiment

* test_hypothesis_emb_variance.py : to find the value of sigma_t (final_std) 
  and store token embeddings
  
* test_hypothesis_emb_variance_2.py : to proceed to statistical testing on stored token embeddings

* find_optimal_alpha.py : performs several noising experiments with various values for alpha

* test_optimal_alpha.py : determines the best alpha based on results of find_optimal_alpha.py

### Running the actual noising experiment

* final_noising_experiment.py : performs the noising experiment *(JSONL)*

### Analyzing results of the noising experiment

* analysis_final_noising_experiment.py : analyses global results of final_noising_experiment.py ; 
  distribution of corrupted biases and relevances, distribution of biases and relevance differences,
  analysis of crucial words

* analysis_distribution_bias_diff.py : analyses the distribution of bias difference, in order to 
  determine an occurrence threshold for the stereotypical score

* analysis_stereotypical_scores.py : analyses stereotypical scores

* looking_for_contexts.py : a util file enabling to filter contexts
  respecting various conditions, and print them out
  
* get_results_for_latex.py : produces a latex table of contexts 
  with inference results, stereotypical scores and bias differences
  
## Related words

* create_dataset_to_parse.py : creates the .txt file that will be parsed *(gitignored !)*

* parsing_script.sh : performs parsing and generates an output file *(gitognored !)* 
  Depending on the size of the dataset to parse, parsing should be split.

* format_parsed_contexts.py : includes parsing results in the context dataset 
  that has been parsed *(JSONL)*

* (find_related_words_utils.py)

* find_related_words.py : adds the list of related words for each context 
  of a given dataset.
  
* check_related_words.py : prints contexts and their related words

* analysis_related_words.py : analyses related words

* analyses_bias_composition.py : tests several estimates of the bias of a context 
  based on the stereotypical scores of the words composing it

## Locating bias (causal tracing)

* create_known_datasets.py *(gitignored !)* : creates the dataset used for causal tracing

* (causal_trace.py / gender_trace.py / causal_tracing_globals.py /
  causal_tracing_utils.py / gender_trace.py / knowns.py)
  Because this code is copied, with slight modifications, 
  from Tomasz Limisiewicz's [DAMA](https://github.com/tomlimi/DAMA) code, 
  the conventions may be different from the rest of the code,
  and thus it should not interfere with the rest of the code.
  
* trace.py : the actual causal tracing experiment
  (several scripts, depending on the dataset used to perform causal tracing)

* (plotting_utils.py)

* plotting.py : plots results

* trace_dataset_analysis.py : provides various statistics 
  about the causal tracing dataset

## Finetuning

* create_datasets_for_dama_finetuning.py : creates datasets for finetuning 
  based on professions dataset, according to various parameters
  (in particular strategies)

* create_datasets_for_finetuning.py : creates a dataset for finetuning
  (ie. with relevant sentences, by adding them a pronoun at the end :
  6 possible strategies are used)
  
* peft_finetuning_stereotyped_related.py / peft_finetuning_stereotyped.py 
  / peft_finetuning_professions.py : performs the actual finetunings
  
* pushing_merged_models_to_hub.py : merging the model and a PEFT module 
  into a single model. Ideally, we would like to avoid using merged models, 
  because it costs additional time and memory (for pushing and loading).
  
* DAMA finetuning is performed using [DAMA](https://github.com/tomlimi/DAMA) 
  and results are reorganized and stored in DAMA_L/ folder
  
## Evaluation

* standard evaluation measures (for bias and language modeling) 
  are computed using 
  [DAMA](https://github.com/tomlimi/DAMA) code
  (after adapting it to PEFT models)

* evaluate_bias_relevance.py : generates inference on test finetuning dataset, for several models

* table_from_bias_relevance.py : generates the table of bias and relevance based on test data inference

* table_he_she.py : prints he/she prediction for some sentences based on test data inference
  
* evaluate_perplexity.py : evaluates the perplexity of several models
  
* evaluation_tasks.py : generates table from evaluation tasks results .json files
  
* evaluation_top_k.py : prints top k prediction for some sentences