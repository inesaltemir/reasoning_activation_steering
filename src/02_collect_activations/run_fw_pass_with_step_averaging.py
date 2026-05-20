"""
Script to run forward passes and extract layer-wise activations for the model in two distinct modes: 'reasoning' and 'baseline'. 
- In 'reasoning' mode, it aligns tokens with logical steps from the ProcessBench dataset to compute mean activation vectors for correct and incorrect reasoning tokens/steps. 
- In 'baseline' mode, it extracts base activations from the FineWeb dataset and computes PCA components to capture baseline language variance.

Type of output:
a `.pt` reasoning-vectors file, where for each layer we store:
  - `mean_correct_token`: global average of every correct reasoning token `[d_model]`
  - `mean_incorrect_token`: global average of every incorrect reasoning token `[d_model]`
  - `reasoning_direction_token`: `mean_correct_token - mean_incorrect_token` `[d_model]`
  - `per_sample_means`: per-sample average activation `[N_samples, d_model]`
  - `mean_correct_samples`: mean activation of perfectly-correct samples `[d_model]`
  - `mean_incorrect_samples`: mean activation of flawed samples `[d_model]`
  - `reasoning_direction_sample`: `mean_correct_samples - mean_incorrect_samples` `[d_model]`
  - `mean_correct_step`:          mean_correct_step,
  - `mean_incorrect_step`:        mean_incorrect_step,
  - `reasoning_direction_step`:   mean_correct_step - mean_incorrect_step,

Note: we do not save the raw activation value for each token, but only global means for (in)correct tokens, steps and samples. 
Structure:
- Arguments: can specifiy '--type reasoning' and '--type baseline'.
- Helper function for reasoning dataset ProcessBench: `prepare_prompt_and_labels_processbench` aligns character offsets to specific reasoning steps to label tokens as correct/incorrect.
- Reasoning mode (`run_reasoning`): use TransformerLens/TransformerBridge to inject custom hooks, extracting and aggregating token-level and step-level means.
- Baseline mode (`run_baseline`): stream the FineWeb dataset via HuggingFace `AutoModelForCausalLM`, extract hidden states, and calculate a 50% variance PCA.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import logging
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA

# ==========================================
# Configuration & Setup
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# Sanitised model name used in folder names and file names (strips org prefix, e.g. "Qwen/Qwen3-8B" → "Qwen3-8B")
MODEL_SLUG = MODEL_NAME.split("/")[-1]

# --- Reasoning-mode config ---
REASONING_DATASET = "ProcessBench"        # dataset folder name, also used in file names
REASONING_DATASET_TAG = REASONING_DATASET.lower()
DATASET_FILE = os.path.join("reasoning_datasets", REASONING_DATASET, "dataset.jsonl")

# --- Baseline-mode config ---
BASELINE_DATASET_TAG = "fineweb"          # short tag used in file names
NUM_SAMPLES = 20000
MAX_LENGTH = 1024
START_TOKEN_IDX = 5
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

# --- Output directory roots ---
LOG_DIR = os.path.join("logs", MODEL_SLUG, REASONING_DATASET_TAG)
VECTOR_DIR = os.path.join("reasoning_vectors", MODEL_SLUG, REASONING_DATASET_TAG)

# Create all output directories up-front so both modes can reference paths safely
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# --- Fully-qualified file paths ---
REASONING_LOG_FILE    = os.path.join(LOG_DIR,    f"reasoning_analysis_{MODEL_SLUG}_{REASONING_DATASET_TAG}_with_steps_avg.log")
REASONING_OUTPUT_FILE = os.path.join(VECTOR_DIR, f"reasoning_vectors_{MODEL_SLUG}_{REASONING_DATASET_TAG}_with_steps_avg.pt")

BASELINE_LOG_FILE    = os.path.join(LOG_DIR,    f"fineweb_baseline_{MODEL_SLUG}_{NUM_SAMPLES}.log")
BASELINE_OUTPUT_FILE = os.path.join(VECTOR_DIR, f"fineweb_activations_{MODEL_SLUG}_{NUM_SAMPLES}.pt")


# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified activation analysis script.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["reasoning", "baseline"],
        help="Analysis mode: 'reasoning' uses ProcessBench + TransformerBridge, "
             "'baseline' uses FineWeb + AutoModelForCausalLM.",
    )
    return parser.parse_args()


# ==========================================
# Logging helper
# ==========================================
def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


# ==========================================
# ProcessBench helpers  (reasoning mode)
# ==========================================
def prepare_prompt_and_labels_processbench(sample, tokenizer):
    """Aligns ProcessBench steps with token positions.
    Bypass usual model.to_tokens(text) usual Transformerlens function because we want character offsets per step to label tokens
    We need to know exactly which token belongs to the "correct" steps and which belongs to the "incorrect" steps
    We must have the offset_mapping (the exact start and end character indices for every single token).

    Returns
    -------
    input_ids : torch.Tensor  shape (1, seq_len)
    token_metadata : list[dict]  per-token {is_correct, step_idx}
    first_error_token_idx : int   (-1 if none)
    reasoning_start_token_idx : int
    """
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [{"start": 0, "end": len(full_text), "step_idx": -1, "is_correct": None}]

    first_error_step_idx = sample["label"]  # -1 means perfectly correct

    for idx, step in enumerate(sample["steps"]):
        start_char = len(full_text)
        step_text = f"Step {idx + 1}: {step}\n"
        full_text += step_text
        end_char = len(full_text)

        # If label is -1, all steps are correct. Otherwise, steps before the label are correct.
        if first_error_step_idx == -1:
            is_correct = True
        else:
            is_correct = True if idx < first_error_step_idx else False

        char_regions.append(
            {"start": start_char, "end": end_char,
             "step_idx": idx, "is_correct": is_correct}
        )

    # Return token IDs and a list of (start, end) character tuples for each step region
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets = encoding["offset_mapping"]

    # Loop through offsets to figure out which reasoning step each token belongs to
    token_metadata = []
    first_error_token_idx = -1      # Default to -1 (no error found)
    reasoning_start_token_idx = -1  # Track where reasoning begins

    for pos, (start, end) in enumerate(offsets):
        # Find which region this token belongs to
        assigned_region = next(
            (r for r in char_regions if start >= r["start"] and end <= r["end"]), None
        )
        if assigned_region:
            step_idx = assigned_region["step_idx"]

            # If this is the first token of Step 0, record the position
            if reasoning_start_token_idx == -1 and step_idx == 0:
                reasoning_start_token_idx = pos

            # If this is the first token that belongs to the error step, record its position
            if (
                first_error_token_idx == -1
                and step_idx == first_error_step_idx
                and first_error_step_idx != -1
            ):
                first_error_token_idx = pos

            token_metadata.append({"is_correct": assigned_region["is_correct"], "step_idx": step_idx})
        else:
            token_metadata.append({"is_correct": None, "step_idx": -1})

    return input_ids, token_metadata, first_error_token_idx, reasoning_start_token_idx


# ==========================================
# REASONING MODE
# ==========================================
def run_reasoning():
    setup_logging(REASONING_LOG_FILE)

    logging.info(f"[reasoning] Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    logging.info(f"Log  → {REASONING_LOG_FILE}")
    logging.info(f"Out  → {REASONING_OUTPUT_FILE}")

    # -- Import here so baseline mode never needs TransformerBridge installed --
    from transformer_lens.model_bridge import TransformerBridge

    # Log GPU
    if torch.cuda.is_available():
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "All (Not Restricted)")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Hardware: GPU {gpu_idx} ({gpu_name})")
    else:
        logging.info("Hardware: CPU")

    logging.info("Loading model via TransformerBridge...")
    model = TransformerBridge.boot_transformers(MODEL_NAME)
    model.enable_compatibility_mode(disable_warnings=True, no_processing=True)
    model.eval()
    d_model = model.cfg.d_model

    resid_post_names = [f"blocks.{layer}.hook_out" for layer in TARGET_LAYERS]
    target_names_set = set(resid_post_names)

    # -- Aggregation tensors --
    running_sum_correct   = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}
    running_sum_incorrect = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}
    
    # New tensors for step-level aggregation
    running_sum_correct_step   = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}
    running_sum_incorrect_step = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}

    per_sample_layer_means      = {name: [] for name in target_names_set}
    per_sample_is_fully_correct = []
    
    count_correct   = 0
    count_incorrect = 0
    count_correct_steps   = 0
    count_incorrect_steps = 0

    # -- Load dataset --
    dataset = []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(dataset)} samples from {DATASET_FILE}.")
    except FileNotFoundError:
        logging.error(f"Dataset file '{DATASET_FILE}' not found. Run format_dataset.py first.")
        return

    # -- Forward passes --
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Forward Passes")):
            try:
                # Tokenize dataset inside prepare_prompt_and_labels
                tokens, token_labels, first_error_token_idx, reasoning_start_token_idx = (
                    prepare_prompt_and_labels_processbench(sample, model.tokenizer)
                )
                # Place token tensors on same device as model
                tokens = tokens.to(model.cfg.device)

                cache = {}

                # Custom hook to ONLY cache reasoning tokens and move them to CPU instantly
                def reasoning_cache_hook(tensor, hook):
                    cache[hook.name] = tensor[:, reasoning_start_token_idx:, :].cpu()

                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(name, reasoning_cache_hook) for name in target_names_set],
                )

                # Slice metadata to perfectly align with our sliced cache
                reasoning_metadata = token_labels[reasoning_start_token_idx:]

                # Track token indices by their step_idx for step-level pooling
                step_to_token_indices = defaultdict(list)

                for pos, meta in enumerate(reasoning_metadata):
                    is_correct = meta["is_correct"]
                    step_idx = meta["step_idx"]
                    
                    if step_idx >= 0:
                        step_to_token_indices[step_idx].append(pos)

                    if is_correct is True:
                        for name in target_names_set:
                            running_sum_correct[name] += cache[name][0, pos]
                        count_correct += 1
                    elif is_correct is False:
                        for name in target_names_set:
                            running_sum_incorrect[name] += cache[name][0, pos]
                        count_incorrect += 1

                # STEP-LEVEL LOGIC:
                label = sample.get("label")
                if label == -1:
                    # Completely correct trajectory: all steps count towards correct_step
                    for s_idx, indices in step_to_token_indices.items():
                        if len(indices) > 0:
                            for name in target_names_set:
                                step_tokens = cache[name][0, indices, :] # shape (num_tokens_in_step, d_model)
                                step_mean = step_tokens.mean(dim=0)
                                running_sum_correct_step[name] += step_mean
                            count_correct_steps += 1
                else:
                    # Incorrect trajectory
                    first_err_idx = label
                    
                    # 1. Process the earlier correct steps (all steps BEFORE first_err_idx)
                    for s_idx in range(first_err_idx):
                        if s_idx in step_to_token_indices:
                            indices = step_to_token_indices[s_idx]
                            if len(indices) > 0:
                                for name in target_names_set:
                                    step_tokens = cache[name][0, indices, :]
                                    step_mean = step_tokens.mean(dim=0)
                                    running_sum_correct_step[name] += step_mean
                                count_correct_steps += 1

                    # 2. Process the first incorrect step
                    if first_err_idx in step_to_token_indices:
                        indices = step_to_token_indices[first_err_idx]
                        if len(indices) > 0:
                            for name in target_names_set:
                                step_tokens = cache[name][0, indices, :]
                                step_mean = step_tokens.mean(dim=0)
                                running_sum_incorrect_step[name] += step_mean
                            count_incorrect_steps += 1

                # Per-sample mean across all reasoning tokens
                for name in target_names_set:
                    sample_avg = cache[name].mean(dim=1).squeeze(0).clone()
                    per_sample_layer_means[name].append(sample_avg)

                # Track if the entire sample's reasoning + final answer was perfectly correct
                is_perfect_sample = (
                    sample.get("label") == -1 and sample.get("final_answer_correct") is True
                )
                per_sample_is_fully_correct.append(is_perfect_sample)

                del cache, logits
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue

    # -- Compute final vectors --
    logging.info("Computing mean vectors...")
    results = {"layers": {}, "metadata": {}}

    is_perfect_mask = torch.tensor(per_sample_is_fully_correct, dtype=torch.bool)

    for name in target_names_set:
        # Token-level means
        mean_correct_token = (
            running_sum_correct[name] / count_correct
            if count_correct > 0 else torch.zeros(d_model)
        )
        mean_incorrect_token = (
            running_sum_incorrect[name] / count_incorrect
            if count_incorrect > 0 else torch.zeros(d_model)
        )

        # Step-level means
        mean_correct_step = (
            running_sum_correct_step[name] / count_correct_steps
            if count_correct_steps > 0 else torch.zeros(d_model)
        )
        mean_incorrect_step = (
            running_sum_incorrect_step[name] / count_incorrect_steps
            if count_incorrect_steps > 0 else torch.zeros(d_model)
        )

        # Sample-level means
        stacked_per_sample_means = torch.stack(per_sample_layer_means[name])

        mean_correct_samples = (
            stacked_per_sample_means[is_perfect_mask].mean(dim=0)
            if is_perfect_mask.any() else torch.zeros(d_model)
        )
        mean_incorrect_samples = (
            stacked_per_sample_means[~is_perfect_mask].mean(dim=0)
            if (~is_perfect_mask).any() else torch.zeros(d_model)
        )

        results["layers"][name] = {
            "mean_correct_token":         mean_correct_token,
            "mean_incorrect_token":       mean_incorrect_token,
            "reasoning_direction_token":  mean_correct_token - mean_incorrect_token,
            
            "mean_correct_step":          mean_correct_step,
            "mean_incorrect_step":        mean_incorrect_step,
            "reasoning_direction_step":   mean_correct_step - mean_incorrect_step,
            
            "per_sample_means":           stacked_per_sample_means,
            "mean_correct_samples":       mean_correct_samples,
            "mean_incorrect_samples":     mean_incorrect_samples,
            "reasoning_direction_sample": mean_correct_samples - mean_incorrect_samples,
        }

    results["metadata"] = {
        "model":                       MODEL_NAME,
        "target_layers":               TARGET_LAYERS,
        "count_correct_tokens":        count_correct,
        "count_incorrect_tokens":      count_incorrect,
        "count_correct_steps":         count_correct_steps,
        "count_incorrect_steps":       count_incorrect_steps,
        "total_successful_samples":    len(per_sample_is_fully_correct),
        "per_sample_is_fully_correct": is_perfect_mask,
    }

    torch.save(results, REASONING_OUTPUT_FILE)
    logging.info(f"Saved reasoning vectors → {REASONING_OUTPUT_FILE}")
    logging.info(f"Token counts  → Correct: {count_correct}, Incorrect: {count_incorrect}")
    logging.info(f"Step counts   → Correct: {count_correct_steps}, Incorrect: {count_incorrect_steps}")
    logging.info(
        f"Sample counts → Perfect: {is_perfect_mask.sum().item()}, "
        f"Flawed: {(~is_perfect_mask).sum().item()}"
    )


# ==========================================
# BASELINE MODE
# ==========================================
def run_baseline():
    setup_logging(BASELINE_LOG_FILE)

    logging.info(f"[baseline] Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    logging.info(f"Log  → {BASELINE_LOG_FILE}")
    logging.info(f"Out  → {BASELINE_OUTPUT_FILE}")

    # -- Import here so reasoning mode never needs full HF stack --
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    logging.info("Loading tokenizer and model via AutoModelForCausalLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    logging.info(f"Loading {DATASET_NAME} (streaming)...")
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    layer_activations = {layer: [] for layer in TARGET_LAYERS}

    count = 0
    with tqdm(total=NUM_SAMPLES, desc="Forward passes") as pbar:
        for row in dataset:
            if count >= NUM_SAMPLES:
                break

            text = row["text"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
            ).to(model.device)

            seq_len = inputs["input_ids"].shape[1]
            if seq_len <= START_TOKEN_IDX + 10:
                continue

            try:
                with torch.no_grad():
                    # output_hidden_states=True gives us the residual stream at every layer
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                for layer in TARGET_LAYERS:
                    # hidden_states[0] = embedding; hidden_states[i] = layer i output
                    layer_hidden = hidden_states[layer]
                    avg_activation = layer_hidden[0, START_TOKEN_IDX:, :].mean(dim=0).cpu()
                    layer_activations[layer].append(avg_activation)

                count += 1
                pbar.update(1)

            except Exception as e:
                logging.error(f"Error processing sample {count}: {e}")
                continue

    logging.info("Stacking and saving activations...")
    results = {
        "layers": {},
        "metadata": {
            "model":           MODEL_NAME,
            "target_layers":   TARGET_LAYERS,
            "num_samples":     count,
            "dataset":         DATASET_NAME,
            "start_token_idx": START_TOKEN_IDX,
        },
    }

    pca_components = {}
    for layer in TARGET_LAYERS:
        stacked = torch.stack(layer_activations[layer])          # (num_samples, d_model)
        results["layers"][layer] = stacked

        X = stacked.to(torch.float32).numpy()
        pca = PCA(n_components=0.50, svd_solver='full')
        pca.fit(X)
        pca_components[layer] = torch.tensor(pca.components_, dtype=torch.float32)
        logging.info(f"  Layer {layer}: {len(pca.components_)} components explain 50% variance.")

    torch.save(results, BASELINE_OUTPUT_FILE)
    logging.info(f"Saved FineWeb activations → {BASELINE_OUTPUT_FILE}")

    pca_output_file = os.path.join(VECTOR_DIR, f"fineweb_pca_components_{MODEL_SLUG}_{NUM_SAMPLES}.pt")
    torch.save({"layers": pca_components, "metadata": results["metadata"]}, pca_output_file)
    logging.info(f"Saved FineWeb PCA components → {pca_output_file}")


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    if args.type == "reasoning":
        run_reasoning()
    else:
        run_baseline()