import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reasoning vector activations across datasets.")
    
    # Input files
    parser.add_argument("--reasoning_file", type=str, required=True, 
                        help="Path to the JSONL file containing reasoning texts (e.g., GPQA).")
    parser.add_argument("--baseline_file", type=str, required=True, 
                        help="Path to the JSONL file containing baseline texts (e.g., FineWeb).")
    parser.add_argument("--vector_file", type=str, required=True, 
                        help="Path to the .pt file containing the candidate reasoning vectors.")
    
    # Data configuration
    parser.add_argument("--reasoning_key", type=str, default="text", 
                        help="The JSON key mapping to the text in the reasoning file.")
    parser.add_argument("--baseline_key", type=str, default="text", 
                        help="The JSON key mapping to the text in the baseline file.")
    parser.add_argument("--num_samples", type=int, default=200, 
                        help="Maximum number of samples to evaluate per dataset.")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", 
                        help="HuggingFace model identifier.")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for model forward passes.")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length for tokenization.")
    
    # Vector configuration
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(18, 29)), 
                        help="List of layers to evaluate. E.g. --layers 16 17 18")
    parser.add_argument("--vector_type", type=str, choices=["step", "sample"], default="step",
                        help="Which vector type to evaluate (if your file has multiple).")
    
    return parser.parse_args()

def load_jsonl_texts(filepath, text_key, max_samples):
    """Loads texts from a JSONL file."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if text_key in data:
                texts.append(data[text_key])
            if len(texts) >= max_samples:
                break
    return texts

def get_activation_scores_robust(model, tokenizer, texts, args, vector_tensor, layer_idx, window_size=4):
    """
    Runs a forward pass and computes robust activation metrics for sparse/bursty vectors.
    """
    scores = {
        "top_5_percentile": [],
        "windowed_max": []
    }
    
    # Ensure vector is normalized
    v_norm = vector_tensor / vector_tensor.norm()
    
    for i in tqdm(range(0, len(texts), args.batch_size), desc=f"Layer {layer_idx}"):
        batch_texts = texts[i:i+args.batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Hidden states shape: [batch, seq_len, d_model]
            hidden_states = outputs.hidden_states[layer_idx]
            
        # Project residual stream onto the normalized vector: shape [batch, seq_len]
        projections = torch.einsum('bsd,d->bs', hidden_states, v_norm)
        mask = inputs["attention_mask"]
        
        for b in range(projections.shape[0]):
            valid_len = mask[b].sum().item()
            
            # Skip highly truncated or empty sequences
            if valid_len < window_size:
                continue 
                
            # Extract non-padded tokens for this sequence
            seq_activations = projections[b, :valid_len]
            
            # 1. Top 5% Percentile (Robust Max)
            k = max(1, int(valid_len * 0.05))
            top_k_values, _ = torch.topk(seq_activations, k)
            scores["top_5_percentile"].append(top_k_values.mean().item())
            
            # 2. Windowed Max (Moving Average)
            seq_tensor = seq_activations.view(1, 1, -1)
            windowed_activations = torch.nn.functional.avg_pool1d(seq_tensor, kernel_size=window_size, stride=1)
            scores["windowed_max"].append(windowed_activations.max().item())
            
    return {k: np.array(v) for k, v in scores.items()}

def compute_metrics(reasoning_scores, baseline_scores):
    """Computes AUROC and Cohen's d between the two distributions."""
    labels = np.concatenate([np.ones(len(reasoning_scores)), np.zeros(len(baseline_scores))])
    all_scores = np.concatenate([reasoning_scores, baseline_scores])
    
    auroc = roc_auc_score(labels, all_scores)
    
    mean_diff = np.mean(reasoning_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt((np.std(reasoning_scores)**2 + np.std(baseline_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return auroc, cohens_d

def main():
    args = parse_args()
    
    print("=== Loading Data ===")
    gpqa_texts = load_jsonl_texts(args.reasoning_file, args.reasoning_key, args.num_samples)
    fineweb_texts = load_jsonl_texts(args.baseline_file, args.baseline_key, args.num_samples)
    
    print(f"Loaded {len(gpqa_texts)} reasoning samples and {len(fineweb_texts)} baseline samples.")
    if len(gpqa_texts) == 0 or len(fineweb_texts) == 0:
        raise ValueError("One or both datasets are empty. Check your file paths and JSON keys.")

    print("\n=== Loading Vectors ===")
    vectors_dict = torch.load(args.vector_file, map_location="cpu")
    # Note: Depending on how your .pt is structured, you might need to adjust this extraction logic.
    # The code below assumes vectors_dict[layer_idx][args.vector_type] yields a 1D tensor [d_model]

    print(f"\n=== Loading Model: {args.model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    model.eval()

    print("\n=== Starting Evaluation ===")
    results = {}
    metrics_to_track = ["top_5_percentile", "windowed_max"]
    
    for layer in args.layers:
        print(f"\n--- Processing Layer {layer} ---")
        
        # Extract the specific vector for this layer
        # Adapt this line based on the exact structure of your saved .pt file
        try:
            target_vector = vectors_dict[layer][args.vector_type].to(model.device, dtype=torch.float16)
        except KeyError:
            print(f"Warning: Layer {layer} or vector type '{args.vector_type}' not found in dict. Skipping.")
            continue
            
        reasoning_scores = get_activation_scores_robust(model, tokenizer, gpqa_texts, args, target_vector, layer)
        baseline_scores = get_activation_scores_robust(model, tokenizer, fineweb_texts, args, target_vector, layer)
        
        results[layer] = {}
        
        for metric in metrics_to_track:
            auroc, cohens_d = compute_metrics(reasoning_scores[metric], baseline_scores[metric])
            results[layer][metric] = {
                "auroc": auroc,
                "effect_size": cohens_d
            }
            print(f"  [{metric}] AUROC: {auroc:.4f} | Cohen's d: {cohens_d:.2f}")

    # Summary: Find best layer based on top_5_percentile AUROC
    best_layer = None
    best_auroc = 0
    
    for layer, metrics in results.items():
        if metrics["top_5_percentile"]["auroc"] > best_auroc:
            best_auroc = metrics["top_5_percentile"]["auroc"]
            best_layer = layer
            
    if best_layer is not None:
        print(f"\n🏆 Optimal Selection: Layer {best_layer} achieved the highest separability (AUROC: {best_auroc:.4f} via Top 5% Metric)")

if __name__ == "__main__":
    main()