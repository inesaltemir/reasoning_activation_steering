import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import logging
import json
from tqdm import tqdm
#from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

# ==========================================
# Configuration & Setup
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"  # target model
TARGET_LAYERS = [18,19,20,21,22,23,24,25,26,27,28]           #  specific layer to analyze, choose subset of layers around 2/3 of model depth
# Qwen3-8B has 36 layers - subset [20-26]
DATASET_FILE = "processbench_formatted_gsm8k.jsonl"
LOG_FILE = "reasoning_analysis.log"
OUTPUT_FILE = "reasoning_vectors_gsm8k.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
#, mode='w'

# ==========================================
# Data processing helpers for ProcessBench dataset
# ==========================================
def prepare_prompt_and_labels_processbench(sample, tokenizer):
    """Aligns ProcessBench steps with token positions
    Bypass usual model.to_tokens(text) usual Transformerlens function because we want character offsets per step to label tokens
    We need to know exactly which token belongs to the "correct" steps and which belongs to the "incorrect" steps
    We must have the offset_mapping (the exact start and end character indices for every single token).
    
    Additionally, must convent to Pytorch tensor and add batch dimension
    """
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [{"start": 0, "end": len(full_text), "step_idx": -1, "is_correct": None}]
    
    first_error_step_idx = sample["label"] # -1 if perfectly correct
    
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
            
        char_regions.append({
            "start": start_char, "end": end_char, 
            "step_idx": idx, "is_correct": is_correct
        })
        
    # Rturn token IDs and a list of (start, end) character tuples for each step region
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets = encoding["offset_mapping"]
    
    # Loop through offsets to figure out which reasoning step each token belongs to
    token_metadata = []
    first_error_token_idx = -1  # Default to -1 (no error found)
    reasoning_start_token_idx = -1  # Track where reasoning begins
    
    for pos, (start, end) in enumerate(offsets):
        # Find which region this token belongs to
        assigned_region = next((r for r in char_regions if start >= r["start"] and end <= r["end"]), None)
        
        if assigned_region:
            step_idx = assigned_region["step_idx"]
            
            # If this is the first token of Step 0, record the position
            if reasoning_start_token_idx == -1 and step_idx == 0:
                reasoning_start_token_idx = pos

            # If this is the first token that belongs to the error step, record its position
            if first_error_token_idx == -1 and step_idx == first_error_step_idx and first_error_step_idx != -1:
                first_error_token_idx = pos
                
            token_metadata.append({
                "is_correct": assigned_region["is_correct"],
                "step_idx": step_idx
            })
        else:
            token_metadata.append({"is_correct": None, "step_idx": -1})
            
    return input_ids, token_metadata, first_error_token_idx, reasoning_start_token_idx

# ==========================================
# Main Execution
# ==========================================
def main():
    logging.info(f"Starting analysis. Model: {MODEL_NAME}, Layer: {TARGET_LAYERS}")
    
    # 1. Load Model
    logging.info("Loading model...")

    # Log the GPU being used
    if torch.cuda.is_available():
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "All (Not Restricted)")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Hardware: Using GPU {gpu_idx} ({gpu_name})")
    else:
        logging.info("Hardware: Running on CPU")

    # Add torch_dtype=torch.bfloat16 for larger models to save VRAM
    # model = HookedTransformer.from_pretrained(MODEL_NAME) is deprecated
    model = TransformerBridge.boot_transformers(MODEL_NAME)
    # model = TransformerBridge.boot_transformers(MODEL_NAME, dtype=torch.bfloat16)

    # Enable compatibility mode for the bridge.
    # This sets up the bridge to work with legacy TransformerLens components/hooks. 
    # It will also disable warnings about the usage of legacy components/hooks if specified.
    # model.enable_compatibility_mode(disable_warnings=True)

    model.enable_compatibility_mode(disable_warnings=True, no_processing=True)

    model.eval() 
    d_model = model.cfg.d_model

    # Iterate and Collect Activations. Legacy alias: hook_resid_post (TransformerLens-2.0)
    resid_post_names = [f"blocks.{layer}.hook_out" for layer in TARGET_LAYERS]
    target_names_set = set(resid_post_names)

    # 2. Initialize Aggregation Tensors on CPU
    running_sum_correct = {name: torch.zeros(d_model, device='cpu') for name in target_names_set}
    running_sum_incorrect = {name: torch.zeros(d_model, device='cpu') for name in target_names_set}
    # Initialize storage for per-sample reasoning averages
    per_sample_layer_means = {name: [] for name in target_names_set}
    per_sample_is_fully_correct = [] # Tracks if the entire sample was perfectly correct
    count_correct = 0
    count_incorrect = 0

    # 3. Load Local Dataset
    dataset = []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(dataset)} samples from {DATASET_FILE}.")
    except FileNotFoundError:
        logging.error(f"Dataset file {DATASET_FILE} not found. Run format_dataset.py first.")
        return

 
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Forward Passes")):
            try:
                # Tokenize dataset inside prepare_prompt_and_labels
                tokens, token_labels, first_error_token_idx, reasoning_start_token_idx = prepare_prompt_and_labels_processbench(sample, model.tokenizer)
                # Place token tensors on same device as model
                tokens = tokens.to(model.cfg.device)
                
                cache = {}
                
                # Custom hook to ONLY cache reasoning tokens and move them to CPU instantly
                def reasoning_cache_hook(tensor, hook):
                    cache[hook.name] = tensor[:, reasoning_start_token_idx:, :].cpu()
                
                # Run forward pass with hooks instead of run_with_cache
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(name, reasoning_cache_hook) for name in target_names_set]
                )
    
                # Slice metadata to perfectly align with our sliced cache
                reasoning_metadata = token_labels[reasoning_start_token_idx:]
    
                # Aggregate vectors using the new relative positions
                for pos, meta in enumerate(reasoning_metadata):
                    is_correct = meta["is_correct"]
                    
                    if is_correct is True:
                        for name in target_names_set:
                            # Tensors are already on CPU from the hook
                            running_sum_correct[name] += cache[name][0, pos]
                        count_correct += 1 
                        
                    elif is_correct is False:
                        for name in target_names_set:
                            running_sum_incorrect[name] += cache[name][0, pos]
                        count_incorrect += 1

                # Compute per-sample average across all reasoning tokens for this sample
                for name in target_names_set:
                    # cache[name] shape is (1, num_reasoning_tokens, d_model)
                    # mean(dim=1) averages across tokens, squeeze(0) removes batch dim
                    sample_avg = cache[name].mean(dim=1).squeeze(0).clone()
                    per_sample_layer_means[name].append(sample_avg)

                # Track if the entire sample's reasoning + final answer was perfectly correct
                is_perfect_sample = (sample.get("label") == -1) and (sample.get("final_answer_correct") is True)
                per_sample_is_fully_correct.append(is_perfect_sample)

                # Free memory immediately
                del cache
                del logits
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue

    # 5. Compute Final Vectors & Save Results
    logging.info("Computing mean vectors...")
    
    results = {"layers": {}, "metadata": {}}

    # Convert the sample-level labels into a boolean tensor mask
    is_perfect_mask = torch.tensor(per_sample_is_fully_correct, dtype=torch.bool)

    
    for name in target_names_set:
        mean_correct = running_sum_correct[name] / count_correct if count_correct > 0 else torch.zeros(d_model)
        mean_incorrect = running_sum_incorrect[name] / count_incorrect if count_incorrect > 0 else torch.zeros(d_model)
        
        # Stack the list into a tensor of shape (num_samples, d_model)
        stacked_per_sample_means = torch.stack(per_sample_layer_means[name])
        
        # Calculate the new dataset averages based on sample-level correctness
        if is_perfect_mask.any():
            mean_fully_correct_samples = stacked_per_sample_means[is_perfect_mask].mean(dim=0)
        else:
            mean_fully_correct_samples = torch.zeros(d_model)
            
        if (~is_perfect_mask).any():
            mean_flawed_samples = stacked_per_sample_means[~is_perfect_mask].mean(dim=0)
        else:
            mean_flawed_samples = torch.zeros(d_model)
        
        results["layers"][name] = {
            "mean_correct": mean_correct, #token wise
            "mean_incorrect": mean_incorrect, #token wise
            "reasoning_direction": mean_correct - mean_incorrect,
            "per_sample_means": stacked_per_sample_means,
            "mean_fully_correct_samples": mean_fully_correct_samples,
            "mean_flawed_samples": mean_flawed_samples 
        }


    results["metadata"] = {
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "count_correct_tokens": count_correct,
        "count_incorrect_tokens": count_incorrect,
        "total_successful_samples": len(per_sample_is_fully_correct),
        # Document the boolean label for every sample, perfectly aligned with 'per_sample_means'
        "per_sample_is_fully_correct": is_perfect_mask 
    }
    
    # RESULTS I HAVE:
    # PER-TOKEN, separated by correct and incorrect tokens, average per-layer of residual stream activation vector
    # mean 
    torch.save(results, OUTPUT_FILE)
    logging.info(f"Successfully saved reasoning vectors to {OUTPUT_FILE}.")
    logging.info(f"Token counts -> Correct: {count_correct}, Incorrect: {count_incorrect}")
    logging.info(f"Sample counts -> Perfect: {is_perfect_mask.sum().item()}, Flawed: {(~is_perfect_mask).sum().item()}")
              
    torch.save(results, OUTPUT_FILE)
    logging.info(f"Successfully saved reasoning vectors to {OUTPUT_FILE}.")
    logging.info(f"Token counts -> Correct: {count_correct}, Incorrect: {count_incorrect}")

if __name__ == "__main__":
    main()