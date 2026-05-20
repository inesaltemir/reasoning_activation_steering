# Reasoning Mechanistic Interpretability work - Inés Altemir 

## Repository structure :books:
-- **src** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- **01_dataset_processing** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```visualize_processbench.py```: Helper functions to visualize ProcessBench samples with erroneous steps. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```format_processbench_dataset.py```: Script to load and format ProcessBench dataset from HuggingFace. Have a dataset with {"problem", "steps", "label" and "final_answer_correct"}.  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```download_deepmind_math.ipynb```: Script to load and format DeepMind Math dataset. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- **02_collect_activations** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```run_fw_pass_with_step_averaging.py```: Script to run forward passes and extract layer-wise activations for the model. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```run_fw_pass_with_step_averaging_storage.py```: Script to run forward passes and extract layer-wise activations for the model, while also storing all raw activation values. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```compute_fw_pass_and_pca_baseline.py```: Script to run forward passes and extract layer-wise activations for the model on "baseline" datasets (FineWeb, DeepMind Math) and compute a joint PCA representation of the averaged-per-sample-per-layer hidden state activations. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```compute_baseline_norms.py```: Script to compute the average L2 norms of baseline model activations, useful for steering experiments later on. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- **03_analyse_reasoning_vectors** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```merge.py```: Remote reindex merge script for Elasticsearch. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```merge.sh```: Slurm script to run merge operation. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- **04_validation_exp** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```correlational_analysis.py```: Script to perform correlational analysis of a chosen "reasoning" vector with "high"-reasoning and "low"-reasoning prompts. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- **05_steering_exp** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```causal_steering.py```: Draft of a script to perform steering experiments. \

