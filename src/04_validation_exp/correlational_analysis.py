import os
import argparse
import json


# At a high level, it passes different types of prompts ("Very-Reasoning" vs. "Null-Reasoning") through a model 
# and measures how closely the model's internal hidden states align with pre-calculated "reasoning vectors" 
# right before it starts generating an answer.

# boundary_activations captures the model's exact internal state after it has read and processed the entire question, 
# but right before it has generated a single word of the answer.

# The script is testing a specific hypothesis: Can we detect if the model is "preparing to reason" 
# by looking at its brain state at the absolute boundary between reading and writing?

# ==========================================
# 1. Parse Arguments FIRST
# ==========================================
parser = argparse.ArgumentParser(description="Run correlational analysis on multiple reasoning vectors.")
parser.add_argument("--gpu", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0', '0,1')")
parser.add_argument("--eval_dataset", type=str, required=True, help="Path to the evaluation dataset (.jsonl)")
parser.add_argument("--vector_file", type=str, required=True, help="Path to the reasoning vectors file (.pt)")
parser.add_argument("--target_layers", type=int, nargs="+", default=list(range(18, 29)), help="List of target layers (e.g. 18 19 20 ... 28)")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="HuggingFace model name")
parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
parser.add_argument("--output_file", type=str, default="/home/ines/Reasoning-activations/results/validation_exp/correlational_results.json", help="Where to save the output JSON")
parser.add_argument("--plot_file", type=str, default="/home/ines/Reasoning-activations/results/validation_exp/correlational_plots.png", help="Where to save the output plot PNG")

# Parse arguments BEFORE any PyTorch imports
args = parser.parse_args()

# ==========================================
# 2. Set CUDA Environment Variables
# ==========================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==========================================
# 3. NOW Import Deep Learning Libraries
# ==========================================
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# The specific vectors you want to test
TARGET_VECTORS = [
    'reasoning_direction_token',
    'reasoning_direction_sample',
    'reasoning_direction_token_cleaned',
    'reasoning_direction_sample_cleaned'
]

def main(args):
    print(f"=== Starting Correlational Analysis ===")
    print(f"GPUs active:   {os.environ['CUDA_VISIBLE_DEVICES']} (Detected {torch.cuda.device_count()} device(s))")
    print(f"Model:         {args.model_name}")
    print(f"Dataset:       {args.eval_dataset}")
    print(f"Vectors:       {args.vector_file}")
    print(f"Layers:        {args.target_layers}")
    print(f"Target Vectors:{TARGET_VECTORS}")
    
    # ==========================================
    # Setup Model & Tokenizer
    # ==========================================
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    model.eval()

    # ==========================================
    # Load Extracted Vectors
    # ==========================================
    print("Loading target reasoning vectors...")
    vector_data = torch.load(args.vector_file, map_location="cpu")
    
    if "layers" not in vector_data:
        raise ValueError(f"Expected top-level 'layers' key in {args.vector_file}")
    
    layers_dict = vector_data["layers"]
    reasoning_vectors = {layer: {} for layer in args.target_layers}
    
    for layer in args.target_layers:
        layer_key = f"blocks.{layer}.hook_out"
        
        if layer_key not in layers_dict:
            raise ValueError(f"Layer key '{layer_key}' not found in {args.vector_file}")
        
        for v_name in TARGET_VECTORS:
            if v_name not in layers_dict[layer_key]:
                print(f"Warning: '{v_name}' not found in {layer_key}. Skipping this vector.")
                continue
                
            # Extract and move to GPU
            vec = layers_dict[layer_key][v_name].to(device=model.device, dtype=torch.bfloat16)
            reasoning_vectors[layer][v_name] = vec

    # ==========================================
    # Load and Format Evaluation Data
    # ==========================================
    print("Preparing dataset...")
    samples = []
    with open(args.eval_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            messages = [{"role": "user", "content": data["problem"]}]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            is_null_reasoning = any(x in data["problem_id"] for x in ["FACT", "TRANS", "CHAT", "EXTR"])
            samples.append({
                "id": data["problem_id"],
                "category": "Null-Reasoning" if is_null_reasoning else "Very-Reasoning",
                "prompt_str": prompt_str
            })

    # ==========================================
    # Forward Pass & Extraction Loop
    # ==========================================
    results = {
        layer: {
            v_name: {"Very-Reasoning": [], "Null-Reasoning": []} 
            for v_name in reasoning_vectors[layer].keys()
        } 
        for layer in args.target_layers
    }

    print(f"\nRunning batched inference (Batch size: {args.batch_size})...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(samples), args.batch_size)):
            batch_samples = samples[i : i + args.batch_size]
            batch_texts = [s["prompt_str"] for s in batch_samples]
            batch_categories = [s["category"] for s in batch_samples]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)

            for layer in args.target_layers:
                layer_hidden_states = outputs.hidden_states[layer]
                boundary_activations = layer_hidden_states[:, -1, :] # Pre-generation boundary
                # check and decode to see what token it actually is, maybe should not only look at this one
                # It grabs only the activation corresponding to the very last token in the sequence.

                for v_name, target_vec in reasoning_vectors[layer].items():
                    target_vec_expanded = target_vec.unsqueeze(0)
                    cos_sims = F.cosine_similarity(boundary_activations, target_vec_expanded, dim=-1)
                    
                    for j, category in enumerate(batch_categories):
                        results[layer][v_name][category].append(cos_sims[j].item())

    # ==========================================
    # Print Terminal Analysis
    # ==========================================
    print("\n=== Correlational Analysis Results (Cosine Similarity) ===")
    for layer in sorted(args.target_layers):
        print(f"\n[ Layer {layer:02d} ]")
        for v_name in reasoning_vectors[layer].keys():
            vr_scores = results[layer][v_name]["Very-Reasoning"]
            nr_scores = results[layer][v_name]["Null-Reasoning"]
            
            vr_mean = np.mean(vr_scores) if vr_scores else 0.0
            nr_mean = np.mean(nr_scores) if nr_scores else 0.0
            print(f"  {v_name:<36} | VR: {vr_mean:+.4f} | NR: {nr_mean:+.4f} | Diff: {vr_mean - nr_mean:+.4f}")

    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed scores to {args.output_file}")

    # ==========================================
    # Generate and Save Plot
    # ==========================================
    print(f"Generating plot...")
    # Create a 2x2 grid of subplots for the 4 vectors
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    sorted_layers = sorted(args.target_layers)

    for idx, v_name in enumerate(TARGET_VECTORS):
        ax = axes[idx]
        vr_means, nr_means = [], []
        
        for layer in sorted_layers:
            if v_name in results[layer]:
                vr_m = np.mean(results[layer][v_name]["Very-Reasoning"])
                nr_m = np.mean(results[layer][v_name]["Null-Reasoning"])
            else:
                vr_m, nr_m = 0.0, 0.0
                
            vr_means.append(vr_m)
            nr_means.append(nr_m)

        # Plot lines
        ax.plot(sorted_layers, vr_means, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Very-Reasoning (VR)')
        ax.plot(sorted_layers, nr_means, marker='x', linestyle='--', color='#d62728', linewidth=2, label='Null-Reasoning (NR)')
        
        # Formatting
        ax.set_title(v_name.replace("_", " ").title(), fontsize=14, fontweight='bold')
        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
        ax.set_xticks(sorted_layers)
        # ax.grid(True, linestyle=':', alpha=0.7)

        ax.grid(True, which="major", alpha=0.6, color="gray", linestyle="-", lw=0.6)
        ax.grid(True, which="minor", alpha=0.2, color="gray", linestyle="--", lw=0.5)
            
        # Add legend only to the first subplot to keep it clean
        if idx == 0:
            ax.legend(fontsize=12, loc='best')

    plt.suptitle("Reasoning Vector Activations at Pre-Generation Boundary", fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to fit the suptitle
    
    plt.savefig(args.plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.plot_file}")

if __name__ == "__main__":
    main(args)