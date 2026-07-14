"""
eval_mean_diff_cosine.py
========================
Evaluate mean-difference steering vectors (from mean_difference_vectors_prm800k.py)
as cosine-similarity direction classifiers.

Two eval-dataset modes (auto-detected by file extension):

  .pt   — Pre-computed step-averaged activation file produced by
           run_fw_pass_with_step_averaging_storage_v2_prm800k.py
           (reasoning_vectors_Qwen3-8B_*_with_steps_avg_storage.pt).
           Labels come directly from the internal per_sample_is_fully_correct
           BoolTensor — no --eval_labels needed or used.
           No model forward pass needed.

  .jsonl — Raw text prompts. Model is loaded and activations are extracted
           on-the-fly. --eval_labels required (one per file).

Usage examples:

  # .pt eval datasets — labels inferred from is_fully_correct mask:
  python eval_mean_diff_cosine.py \
      --vectors mean_diff_vectors_prm800k.pt \
      --eval_datasets reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt

  # multiple .pt files — each split internally by is_fully_correct:
  python eval_mean_diff_cosine.py \
      --vectors mean_diff_vectors_prm800k.pt \
      --eval_datasets reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt \
                      reasoning_vectors_Qwen3-8B_math_shepherd_with_steps_avg_storage.pt

  # .jsonl eval datasets (GPU needed):
  python eval_mean_diff_cosine.py \
      --vectors mean_diff_vectors_prm800k.pt \
      --eval_datasets reasoning_eval.jsonl non_reasoning_hard.jsonl \
      --eval_labels   reasoning non_reasoning_hard \
      --model_name    Qwen/Qwen3-8B
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--vectors", default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/mean_diff_vectors_prm800k.pt",
                    help="Path to mean_diff_vectors_prm800k.pt")
parser.add_argument("--eval_datasets", nargs="+", required=True,
                    help="Eval dataset paths: .pt (pre-computed) or .jsonl (raw text)")
parser.add_argument("--eval_labels", nargs="+", default=None,
                    help=".jsonl mode only: label per file. Not used for .pt files "
                         "(labels come from the internal is_fully_correct mask).")
parser.add_argument("--output_dir", default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/eval_mean_diff_results")
parser.add_argument("--target_layers", type=int, nargs="+", default=None,
                    help="Subset of integer layer indices to evaluate (default: all)")
# Only relevant for .jsonl mode:
parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
parser.add_argument("--gpu", default="0")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--token_positions", default="mean_all",
                    choices=["last", "mean_all"])
args = parser.parse_args()

# Detect mode
all_pt = all(p.endswith(".pt")    for p in args.eval_datasets)
all_jl = all(p.endswith(".jsonl") for p in args.eval_datasets)
if not (all_pt or all_jl):
    print("ERROR: Mix of .pt and .jsonl eval datasets is not supported.")
    sys.exit(1)

# For .jsonl mode, eval_labels is required and must match dataset count
if all_jl:
    if args.eval_labels is None:
        print("ERROR: --eval_labels is required for .jsonl eval datasets.")
        sys.exit(1)
    if len(args.eval_datasets) != len(args.eval_labels):
        print("ERROR: --eval_datasets and --eval_labels must have the same length.")
        sys.exit(1)

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load steering vectors
# ---------------------------------------------------------------------------
print(f"Loading steering vectors from {args.vectors} ...")
raw = torch.load(args.vectors, map_location="cpu", weights_only=False)
# Structure: raw["layers"]["blocks.18.hook_out"]["steering_vector"] -> Tensor[d_model]

def hook_to_layer_int(hook_name: str) -> int | None:
    m = re.search(r"(\d+)", hook_name)
    return int(m.group(1)) if m else None

vectors = {}
for hook_name, layer_data in raw["layers"].items():
    layer_int = hook_to_layer_int(hook_name)
    if layer_int is None:
        continue
    if args.target_layers and layer_int not in args.target_layers:
        continue
    vectors[layer_int] = layer_data["steering_vector"].float()

available_layers = sorted(vectors.keys())
print(f"  Loaded {len(vectors)} steering vectors — layers: {available_layers}")

# ---------------------------------------------------------------------------
# Discriminability metrics
# ---------------------------------------------------------------------------
def compute_discriminability(pos: np.ndarray, neg: np.ndarray) -> dict:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return {"cohens_d": 0.0, "auroc": 0.5, "selectivity_ratio": 0.0,
                "mann_whitney_p": 1.0, "mean_pos": 0.0, "mean_neg": 0.0,
                "gap": 0.0, "n_pos": len(pos), "n_neg": len(neg)}
    n_pos, n_neg = len(pos), len(neg)
    mean_pos, mean_neg = np.mean(pos), np.mean(neg)
    std_pos = np.std(pos, ddof=1)
    std_neg = np.std(neg, ddof=1)
    pooled  = np.sqrt(((n_pos-1)*std_pos**2 + (n_neg-1)*std_neg**2) / (n_pos+n_neg-2))
    cohens_d = (mean_pos - mean_neg) / (pooled + 1e-10)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos, neg])
    try:    auroc = roc_auc_score(labels, scores)
    except: auroc = 0.5
    sel = mean_pos / (abs(mean_neg) + 1e-8)
    try:    _, p = stats.mannwhitneyu(pos, neg, alternative="greater")
    except: p = 1.0
    return {"cohens_d": cohens_d, "auroc": auroc, "selectivity_ratio": sel,
            "mann_whitney_p": p, "mean_pos": mean_pos, "mean_neg": mean_neg,
            "gap": mean_pos - mean_neg, "n_pos": n_pos, "n_neg": n_neg}

# ---------------------------------------------------------------------------
# PATH A: .pt pre-computed activations
# ---------------------------------------------------------------------------
# scores[layer]["correct" | "incorrect"] = list of cosine sim floats
# Each .pt file is split internally using data["metadata"]["per_sample_is_fully_correct"].
# Multiple .pt files are pooled together (useful to compare datasets or increase N).

def eval_from_pt() -> dict[int, dict[str, list]]:
    """
    For each .pt file:
      - Load per_sample_means [N, d_model] (one row per sample, per layer hook).
      - Load per_sample_is_fully_correct BoolTensor[N] from metadata.
      - Split cosine similarities into "correct" / "incorrect" lists accordingly.
    Multiple files are pooled: scores are accumulated across all input .pt files.
    """
    scores: dict[int, dict[str, list]] = {l: {"correct": [], "incorrect": []} for l in available_layers}

    for path in args.eval_datasets:
        dataset_name = Path(path).stem
        print(f"  Loading '{dataset_name}' ...")
        data = torch.load(path, map_location="cpu", weights_only=False)

        # is_correct mask: BoolTensor[N], True = fully correct sample
        is_correct: torch.Tensor = data["metadata"]["per_sample_is_fully_correct"]
        n_correct   = is_correct.sum().item()
        n_incorrect = (~is_correct).sum().item()
        print(f"    {n_correct} correct, {n_incorrect} incorrect samples")

        for hook_name, layer_data in data["layers"].items():
            layer_int = hook_to_layer_int(hook_name)
            if layer_int not in vectors:
                continue

            per_sample_means = layer_data["per_sample_means"].float()  # [N, d_model]
            direction = vectors[layer_int]                              # [d_model]

            cos_sims = F.cosine_similarity(per_sample_means, direction.unsqueeze(0), dim=-1)
            # [N] cosine similarities

            scores[layer_int]["correct"].extend(  cos_sims[is_correct].tolist())
            scores[layer_int]["incorrect"].extend(cos_sims[~is_correct].tolist())

    return scores

# ---------------------------------------------------------------------------
# PATH B: .jsonl raw text → model forward pass
# ---------------------------------------------------------------------------
def eval_from_jsonl() -> dict[int, dict[str, list]]:
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = model.device

    vecs_gpu = {l: v.to(device=device, dtype=torch.bfloat16) for l, v in vectors.items()}

    all_labels = args.eval_labels
    scores: dict[int, dict[str, list]] = {l: defaultdict(list) for l in available_layers}

    for label, path in zip(all_labels, args.eval_datasets):
        print(f"  Loading '{label}' from {path} ...")
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                text = d.get("problem") or d.get("text") or d.get("prompt") or d.get("content", "")
                if text:
                    samples.append(text)
        print(f"    {len(samples)} samples — running forward pass ...")

        with torch.inference_mode():
            for i in tqdm(range(0, len(samples), args.batch_size), desc=f"  {label}"):
                batch  = samples[i : i + args.batch_size]
                inputs = tokenizer(batch, return_tensors="pt",
                                   padding=True, truncation=True).to(device)
                out = model(**inputs, output_hidden_states=True)

                for layer_int, direction in vecs_gpu.items():
                    hs   = out.hidden_states[layer_int]        # [B, S, d_model]
                    mask = inputs["attention_mask"]             # [B, S]

                    if args.token_positions == "last":
                        activations = hs[:, -1, :]
                    else:  # mean_all
                        m = mask.unsqueeze(-1).float()
                        activations = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

                    cos_sims = F.cosine_similarity(
                        activations.float(), direction.float().unsqueeze(0), dim=-1
                    )
                    scores[layer_int][label].extend(cos_sims.tolist())

    return scores

# ---------------------------------------------------------------------------
# Run eval
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
if all_pt:
    print("Mode: pre-computed .pt activations — labels from is_fully_correct")
    scores = eval_from_pt()
    pos_label = "correct"
    neg_labels = ["incorrect"]
else:
    print("Mode: .jsonl raw text (model forward pass)")
    scores = eval_from_jsonl()
    # Identify positive label (first label with "reason" but not "non")
    pos_label = next(
        (l for l in args.eval_labels if "reason" in l.lower() and "non" not in l.lower()),
        args.eval_labels[0]
    )
    neg_labels = [l for l in args.eval_labels if l != pos_label]

print(f"Positive: '{pos_label}', Negatives: {neg_labels}")

# ---------------------------------------------------------------------------
# Compute & print metrics
# ---------------------------------------------------------------------------
metrics = {}
for layer in available_layers:
    pos_scores = np.array(scores[layer][pos_label])
    for neg_label in neg_labels:
        neg_scores = np.array(scores[layer][neg_label])
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        metrics[(layer, neg_label)] = compute_discriminability(pos_scores, neg_scores)

print(f"\n{'='*90}")
print("  MEAN-DIFF VECTOR — DIRECTION (COSINE) EVALUATION SUMMARY")
print(f"{'='*90}")

best_auroc, best_layer = -1, None
for neg_label in neg_labels:
    print(f"\n  vs '{neg_label}'")
    print(f"  {'Layer':>6} | {'AUROC':>8} | {'Cohen d':>8} | {'gap':>8} | {'mean+':>8} | {'mean-':>8}")
    print(f"  {'-'*62}")
    for layer in available_layers:
        m = metrics.get((layer, neg_label), {})
        auroc = m.get("auroc", 0.5)
        print(f"  {layer:>6} | {auroc:>8.4f} | {m.get('cohens_d',0):>+8.4f} | "
              f"{m.get('gap',0):>+8.4f} | {m.get('mean_pos',0):>8.4f} | {m.get('mean_neg',0):>8.4f}")
        if auroc > best_auroc:
            best_auroc, best_layer = auroc, (layer, neg_label)

print(f"\n{'='*90}")
if best_layer:
    print(f"  ★  Best: layer {best_layer[0]} vs '{best_layer[1]}'  "
          f"AUROC={best_auroc:.4f}, Cohen's d={metrics[best_layer].get('cohens_d',0):+.4f}")
print(f"{'='*90}\n")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def safe_float(x):
    f = float(x)
    return None if (np.isnan(f) or np.isinf(f)) else f

all_labels_in_scores = [pos_label] + neg_labels

out = {"direction_metrics": {}, "raw_scores": {}}

for (layer, neg_label), m in metrics.items():
    out["direction_metrics"][f"layer{layer}__vs_{neg_label}"] = {
        k: safe_float(v) if isinstance(v, (float, np.floating)) else v
        for k, v in m.items()
    }

for layer in available_layers:
    for label in all_labels_in_scores:
        arr = scores[layer].get(label, [])
        if arr:
            out["raw_scores"][f"layer{layer}__{label}"] = {
                "cosine_mean": safe_float(np.mean(arr)),
                "cosine_std":  safe_float(np.std(arr)),
                "n":           len(arr),
            }

results_path = os.path.join(args.output_dir, "eval_mean_diff_cosine.json")
with open(results_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved → {results_path}")