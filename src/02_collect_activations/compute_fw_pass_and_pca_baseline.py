"""
Script to extract baseline hidden state activations from multiple distinct "baseline" datasets (FineWeb, DeepMind Math) 
and compute a joint PCA representation of the averaged per sample per layer hidden state activations. 
Purpose: to create a robust and holistic baseline representation, 
useful for deconfounding general text processing from true reasoning activations.

Structure:
- Activation extraction & caching: run HF forward passes to collect average hidden states per layer, and skipping if already cached.
- Joint PCA computation: Loads multiple cached activation files, concatenates them along the sample dimension for each layer, and computes/saves a joint PCA.
"""


import os
import argparse
import torch
import logging
import json
from tqdm import tqdm
from sklearn.decomposition import PCA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# Configuration & Global Constants
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_SLUG = MODEL_NAME.split("/")[-1]
TARGET_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

MAX_LENGTH = 1024
START_TOKEN_IDX = 2             # had used 5 for FineWeb and ProcessBench

VECTOR_DIR = os.path.join("baseline_vectors", MODEL_SLUG)
LOG_DIR = os.path.join("logs", MODEL_SLUG)

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# File paths for specific local datasets
LOCAL_DATASETS = {
    "deepmind_math": "/home/ines/Reasoning-activations/reasoning_datasets/DeepMind-Math_Dataset/formatted_math_samples.jsonl"
}

# ==========================================
# Logging Helper
# ==========================================
def setup_logging(log_filename: str):
    log_path = os.path.join(LOG_DIR, log_filename)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

# ==========================================
# Module 1: Dataset Streamers
# ==========================================
def get_dataset_stream(dataset_tag):
    """Factory function yielding formatted text strings for a given dataset."""
    if dataset_tag == "fineweb":
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        for row in dataset:
            yield row["text"]
            
    elif dataset_tag == "deepmind_math":
        file_path = LOCAL_DATASETS["deepmind_math"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing local dataset file: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                yield f"Question: {data.get('question', '')}\nAnswer: {data.get('answer', '')}"
    else:
        raise ValueError(f"Unknown baseline dataset specified: {dataset_tag}")

# ==========================================
# Module 2: Activation Extraction & Caching
# ==========================================
def extract_activations_for_dataset(dataset_tag, num_samples, force_recompute=False):
    """
    Checks for cached activations. If none exist (or forced), runs the forward 
    pass over the dataset to extract and save the target layers' activations.
    Returns the path to the saved activations.
    """
    output_file = os.path.join(VECTOR_DIR, f"{dataset_tag}_activations_{num_samples}.pt")
    
    if os.path.exists(output_file) and not force_recompute:
        logging.info(f"⏭️ Cache found for {dataset_tag} ({num_samples} samples). Skipping forward passes.")
        return output_file
        
    logging.info(f"▶️ No cache found for {dataset_tag}. Initializing model for forward passes...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    layer_activations = {layer: [] for layer in TARGET_LAYERS}
    stream = get_dataset_stream(dataset_tag)
    
    count = 0
    with tqdm(total=num_samples, desc=f"Extracting {dataset_tag}") as pbar:
        for text in stream:
            if count >= num_samples:
                break

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            if inputs["input_ids"].shape[1] <= START_TOKEN_IDX + 10:
                continue

            try:
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    
                for layer in TARGET_LAYERS:
                    avg_activation = outputs.hidden_states[layer][0, START_TOKEN_IDX:, :].mean(dim=0).cpu()
                    layer_activations[layer].append(avg_activation)

                count += 1
                pbar.update(1)
            except Exception as e:
                logging.error(f"Error on sample {count}: {e}")
                continue

    # Free GPU memory immediately
    del model, tokenizer
    torch.cuda.empty_cache()

    # Stack and save
    results = {"layers": {}, "metadata": {"model": MODEL_NAME, "dataset": dataset_tag, "samples": count, "start_token_idx": START_TOKEN_IDX, "target_layers": TARGET_LAYERS}}
    for layer in TARGET_LAYERS:
        results["layers"][layer] = torch.stack(layer_activations[layer])
        
    torch.save(results, output_file)
    logging.info(f"✅ Saved activations to {output_file}")
    
    return output_file

# ==========================================
# Module 3: Joint PCA Computation
# ==========================================
def compute_joint_pca(activation_files, num_samples):
    """
    Loads multiple activation files, concatenates them across datasets for each layer,
    and computes a single Joint PCA representation.
    """
    logging.info("\n" + "="*40)
    logging.info("🧠 Computing Joint PCA over multitude of datasets...")
    
    # Structure to hold concatenated activations: {layer: [tensor1, tensor2, ...]}
    combined_activations = {layer: [] for layer in TARGET_LAYERS}
    total_samples = 0
    
    for filepath in activation_files:
        logging.info(f"Loading {os.path.basename(filepath)}...")
        data = torch.load(filepath, map_location="cpu")
        # total_samples += data["metadata"]["samples"]
        total_samples += data["metadata"].get("samples", data["metadata"].get("num_samples", 0))
        
        for layer in TARGET_LAYERS:
            combined_activations[layer].append(data["layers"][layer])
            
    pca_results = {"layers": {}, "metadata": {
        "model": MODEL_NAME,
        "total_samples": total_samples,
        "datasets_combined": len(activation_files)
    }}
    
    for layer in TARGET_LAYERS:
        # Concatenate along the sample dimension (dim=0)
        layer_X = torch.cat(combined_activations[layer], dim=0).to(torch.float32).numpy()
        
        logging.info(f"Fitting PCA for Layer {layer} on shape {layer_X.shape}...")
        pca = PCA(n_components=0.50, svd_solver='full')
        pca.fit(layer_X)
        
        pca_results["layers"][layer] = torch.tensor(pca.components_, dtype=torch.float32)
        logging.info(f"  ↳ Layer {layer}: {len(pca.components_)} components explain 50% variance.")
        
    # Save the joint PCA
    dataset_tags = "_".join([os.path.basename(f).split('_')[0] for f in activation_files])
    pca_output_file = os.path.join(VECTOR_DIR, f"joint_pca_{dataset_tags}_{total_samples}samples.pt")
    torch.save(pca_results, pca_output_file)
    
    logging.info(f"✅ Joint PCA saved successfully to -> {pca_output_file}")

# ==========================================
# Main Execution Flow
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Modular Activation & PCA Pipeline")
    parser.add_argument("--datasets", nargs="+", default=["fineweb", "deepmind_math"],
                        help="List of baseline datasets to process.")
    parser.add_argument("--num_samples", type=int, default=20000, 
                        help="Number of samples to process per dataset.")
    parser.add_argument("--force_recompute", action="store_true", 
                        help="Ignore cache and force model forward passes.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging("pipeline_execution.log")
    
    logging.info(f"Starting pipeline for datasets: {args.datasets}")
    logging.info(f"Samples per dataset: {args.num_samples}")
    
    # Step 1: Ensure we have activations for all requested datasets
    activation_files = []
    for ds in args.datasets:
        logging.info(f"\n--- Processing Dataset: {ds} ---")
        filepath = extract_activations_for_dataset(ds, args.num_samples, args.force_recompute)
        activation_files.append(filepath)
        
    # Step 2: Compute Joint PCA
    compute_joint_pca(activation_files, args.num_samples)
