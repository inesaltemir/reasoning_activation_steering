import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import logging
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# Configuration & Setup
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"            # Target model
TARGET_LAYERS = [18,19,20,21,22,23,24,25,26,27,28]   
NUM_SAMPLES = 20000                      # Number of Fineweb transcripts to process
MAX_LENGTH = 1024                       # Max sequence length per transcript
START_TOKEN_IDX = 5                    # Token index to start averaging from (Anthropic standard)

DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"          # Using the 10B token sample config for fineweb
LOG_FILE = "logs/fineweb_baseline_20000.log"
OUTPUT_FILE = "reasoning_vectors/Qwen3-8B/fineweb_activations_20000.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def main():
    logging.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We use standard HF here, extracting hidden states is straightforward
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()
    
    logging.info(f"Loading {DATASET_NAME} dataset (streaming mode)...")
    # Streaming prevents downloading terabytes of text
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)
    
    # Dictionary to hold the final averaged activations per layer
    layer_activations = {layer: [] for layer in TARGET_LAYERS}
    
    logging.info(f"Processing {NUM_SAMPLES} samples...")
    
    count = 0
    with tqdm(total=NUM_SAMPLES, desc="Forward passes") as pbar:
        for row in dataset:
            if count >= NUM_SAMPLES:
                break
                
            text = row["text"]
            
            # Tokenize text
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=MAX_LENGTH
            ).to(model.device)
            
            seq_len = inputs["input_ids"].shape[1]
            
            # Skip texts that are too short to be meaningful
            if seq_len <= START_TOKEN_IDX + 10:
                continue 
                
            try:
                with torch.no_grad():
                    # output_hidden_states=True gives us the residual stream at every layer
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states 
                    
                for layer in TARGET_LAYERS:
                    # Note: hidden_states[0] is the embedding layer, hidden_states[1] is layer 1.
                    # Shape: (batch=1, seq_len, d_model)
                    layer_hidden = hidden_states[layer]
                    
                    # Average across tokens, starting from the 50th token
                    avg_activation = layer_hidden[0, START_TOKEN_IDX:, :].mean(dim=0).cpu()
                    
                    layer_activations[layer].append(avg_activation)
                    
                count += 1
                pbar.update(1)
                
            except Exception as e:
                logging.error(f"Error processing sample {count}: {e}")
                continue

    # ==========================================
    # Save Results
    # ==========================================
    logging.info("Concatenating and saving activations...")
    results = {"layers": {}, "metadata": {
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "num_samples": count,
        "dataset": DATASET_NAME,
        "start_token_idx": START_TOKEN_IDX
    }}
    
    for layer in TARGET_LAYERS:
        # Stack into a single tensor of shape: (num_samples, d_model)
        stacked_acts = torch.stack(layer_activations[layer])
        results["layers"][layer] = stacked_acts
        
    torch.save(results, OUTPUT_FILE)
    logging.info(f"Successfully saved Fineweb activations to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()