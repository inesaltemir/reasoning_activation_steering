import torch
import logging
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ==========================================
# Configuration & Setup
# ==========================================
MODEL_NAME = "gpt2-small"  # Change to your target model (e.g., "meta-llama/Llama-2-7b-hf")
TARGET_LAYER = 6           # The specific layer you want to analyze to save VRAM
LOG_FILE = "reasoning_analysis.log"
OUTPUT_FILE = "reasoning_vectors.pt"

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ==========================================
# Data Processing Helpers
# ==========================================
def prepare_prompt_and_labels(sample, tokenizer):
    """Aligns ProcessBench steps with token positions."""
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [{"start": 0, "end": len(full_text), "step_idx": -1, "is_correct": None}]
    
    first_error_idx = sample["label"]
    
    for idx, step in enumerate(sample["steps"]):
        start_char = len(full_text)
        step_text = f"Step {idx + 1}: {step}\n"
        full_text += step_text
        end_char = len(full_text)
        
        is_correct = True if idx < first_error_idx else False
        char_regions.append({
            "start": start_char, "end": end_char, 
            "step_idx": idx, "is_correct": is_correct
        })
        
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets = encoding["offset_mapping"]
    
    token_metadata = []
    for start, end in offsets:
        assigned_region = next((r for r in char_regions if start >= r["start"] and end <= r["end"]), None)
        if assigned_region:
            token_metadata.append({"is_correct": assigned_region["is_correct"]})
        else:
            token_metadata.append({"is_correct": None})
            
    return input_ids, token_metadata

# ==========================================
# Main Execution
# ==========================================
def main():
    logging.info(f"Starting analysis. Model: {MODEL_NAME}, Target Layer: {TARGET_LAYER}")
    
    # 1. Load Model (Using bfloat16 to save VRAM if using a larger model)
    logging.info("Loading model...")
    # NOTE: For Llama/Mistral, add torch_dtype=torch.bfloat16 to from_pretrained
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    d_model = model.cfg.d_model

    # 2. Initialize Online Aggregation Tensors (Saves Disk Space)
    # We store these on CPU to keep VRAM strictly for the model's forward passes
    running_sum_correct = torch.zeros(d_model, device='cpu')
    running_sum_incorrect = torch.zeros(d_model, device='cpu')
    count_correct = 0
    count_incorrect = 0

    # 3. Load Dataset (Mocking ProcessBench dataset here for illustration)
    # In reality, load your JSONL file: `data = [json.loads(line) for line in open('processbench.jsonl')]`
    dataset = [
        {
            "problem": "Pat has a flower bed...",
            "steps": [
                "First, calculate total length...",
                "Second, calculate spaces...", # Error happens here
                "Third, determine cost..."
            ],
            "label": 1 
        }
    ] # Add your dataset items here
    
    logging.info(f"Loaded dataset with {len(dataset)} samples.")
    
    # 4. Iterate and collect
    resid_post_name = f"blocks.{TARGET_LAYER}.hook_resid_post"
    
    # Using torch.no_grad() is critical to prevent VRAM leaks during inference
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Processing Samples")):
            try:
                tokens, token_labels = prepare_prompt_and_labels(sample, model.tokenizer)
                tokens = tokens.to(model.cfg.device)
                
                # run_with_cache with names_filter ensures we ONLY store the layer we need in RAM
                logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == resid_post_name
                )
                
                # Shape: [1, seq_len, d_model] -> [seq_len, d_model]
                layer_activations = cache[resid_post_name][0].cpu() 
                
                # Aggregate
                for pos, meta in enumerate(token_labels):
                    is_correct = meta["is_correct"]
                    if is_correct is True:
                        running_sum_correct += layer_activations[pos]
                        count_correct += 1
                    elif is_correct is False:
                        running_sum_incorrect += layer_activations[pos]
                        count_incorrect += 1
                
                # Explicitly clear cache to free VRAM for the next sequence
                del cache
                del logits
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue

    # 5. Compute Final Vectors & Save
    logging.info("Computing mean vectors...")
    
    mean_correct = running_sum_correct / count_correct if count_correct > 0 else torch.zeros(d_model)
    mean_incorrect = running_sum_incorrect / count_incorrect if count_incorrect > 0 else torch.zeros(d_model)
    
    reasoning_direction = mean_correct - mean_incorrect
    
    results = {
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "reasoning_direction": reasoning_direction,
        "metadata": {
            "model": MODEL_NAME,
            "layer": TARGET_LAYER,
            "count_correct_tokens": count_correct,
            "count_incorrect_tokens": count_incorrect
        }
    }
    
    torch.save(results, OUTPUT_FILE)
    logging.info(f"Successfully saved vectors to {OUTPUT_FILE} (File size is minimal).")
    logging.info(f"Tokens processed - Correct: {count_correct}, Incorrect: {count_incorrect}")

if __name__ == "__main__":
    main()