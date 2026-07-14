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

Optional step-level evaluation (--raw_activations_dir):
  Reads the raw shard files produced alongside the .pt file.
  For each (sample_idx, step_idx) group, computes the mean activation
  across all tokens of that step, then computes cosine similarity with
  the steering vector. Steps are split by their per-token is_correct label.
  This gives a finer-grained view than sample-level means.

Usage examples:

  # .pt eval (sample-level only):
  python eval_mean_diff_cosine.py \
      --vectors mean_diff_vectors_prm800k.pt \
      --eval_datasets reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt

  # .pt eval + step-level eval from raw shards:
  python eval_mean_diff_cosine.py \
      --vectors mean_diff_vectors_prm800k.pt \
      --eval_datasets reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt \
      --raw_activations_dir /path/to/raw_activations

  # .jsonl eval (GPU needed):
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
                    help=".jsonl mode only: label per file. Ignored for .pt files.")
parser.add_argument("--raw_activations_dir", default=None,
                    help="Path to the raw_activations/ directory written alongside the .pt file. "
                         "Required to run step-level evaluation. Only used in .pt mode.")
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

# Also build the reverse: layer_int -> hook_name (needed for shard path construction)
layer_to_hook: dict[int, str] = {}
vectors: dict[int, torch.Tensor] = {}
for hook_name, layer_data in raw["layers"].items():
    layer_int = hook_to_layer_int(hook_name)
    if layer_int is None:
        continue
    if args.target_layers and layer_int not in args.target_layers:
        continue
    vectors[layer_int]     = layer_data["steering_vector"].float()
    layer_to_hook[layer_int] = hook_name

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
# PATH A: .pt pre-computed activations  →  SAMPLE-LEVEL eval
# ---------------------------------------------------------------------------
def eval_from_pt() -> dict[int, dict[str, list]]:
    """
    For each .pt file:
      - Load per_sample_means [N, d_model]: the mean activation across all
        reasoning tokens of each sample, at each layer.
      - Load per_sample_is_fully_correct BoolTensor[N] from metadata.
      - Compute cosine similarity of each sample mean with the steering vector.
      - Split into "correct" / "incorrect" buckets by the mask.
    Multiple .pt files are pooled into the same buckets.
    """
    scores: dict[int, dict[str, list]] = {l: {"correct": [], "incorrect": []} for l in available_layers}

    for path in args.eval_datasets:
        dataset_name = Path(path).stem
        print(f"  Loading '{dataset_name}' ...")
        data = torch.load(path, map_location="cpu", weights_only=False)

        is_correct: torch.Tensor = data["metadata"]["per_sample_is_fully_correct"]
        print(f"    {is_correct.sum().item()} correct, {(~is_correct).sum().item()} incorrect samples")

        for hook_name, layer_data in data["layers"].items():
            layer_int = hook_to_layer_int(hook_name)
            if layer_int not in vectors:
                continue

            per_sample_means = layer_data["per_sample_means"].float()  # [N, d_model]
            direction = vectors[layer_int]                              # [d_model]

            cos_sims = F.cosine_similarity(per_sample_means, direction.unsqueeze(0), dim=-1)

            scores[layer_int]["correct"].extend(  cos_sims[is_correct].tolist())
            scores[layer_int]["incorrect"].extend(cos_sims[~is_correct].tolist())

    return scores


# ---------------------------------------------------------------------------
# PATH A2: raw activation shards  →  STEP-LEVEL eval
# ---------------------------------------------------------------------------
def eval_steps_from_raw(raw_dir: Path) -> dict[int, dict[str, list]]:
    """
    Read the raw shard files in raw_dir to compute per-step cosine similarities.

    Raw shard layout (written by DiskBackedActivationStore):
        raw_dir/
            index.pt                          — sample index + shard counts
            meta_shard_0000.pt ...            — list[dict] per token row:
                                                  {sample_idx, step_idx, is_correct, ...}
            blocks.18.hook_out/
                shard_0000.pt ...             — (N_tokens, d_model) activation rows

    For each layer:
      1. Load all token activations into a flat tensor [T_total, d_model].
      2. Load all metadata rows (same row order).
      3. Group token rows by (sample_idx, step_idx) — one group = one reasoning step.
      4. Average token activations within each group → step mean [d_model].
      5. Determine step label from the majority is_correct vote across its tokens.
      6. Compute cosine similarity of step mean with steering vector.
      7. Accumulate into "correct" / "incorrect" lists.

    Returns scores[layer_int]["correct" | "incorrect"] = list of floats.
    """
    index_path = raw_dir / "index.pt"
    if not index_path.exists():
        raise FileNotFoundError(f"index.pt not found at {index_path}")

    index      = torch.load(index_path, map_location="cpu", weights_only=False)
    num_shards = index["num_shards"]

    # ---- Load all metadata once (shared across layers) ----
    print(f"  Loading metadata from {num_shards} shards ...")
    all_meta: list[dict] = []
    for sid in range(num_shards):
        all_meta.extend(
            torch.load(raw_dir / f"meta_shard_{sid:04d}.pt", weights_only=False)
        )
    print(f"  {len(all_meta)} token rows loaded")

    # ---- Build (sample_idx, step_idx) groups once ----
    # Each group maps to the list of absolute row indices that belong to it.
    # Rows with step_idx < 0 are non-reasoning tokens (prompt, padding) — skip them.
    step_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_idx, meta in enumerate(all_meta):
        step_idx = meta.get("step_idx", -1)
        if step_idx < 0:
            continue  # not a reasoning step token
        sample_idx = meta["sample_idx"]
        step_groups[(sample_idx, step_idx)].append(row_idx)

    print(f"  {len(step_groups)} (sample, step) groups found")

    step_scores: dict[int, dict[str, list]] = {
        l: {"correct": [], "incorrect": []} for l in available_layers
    }

    # ---- Process one layer at a time to keep memory bounded ----
    for layer_int in tqdm(available_layers, desc="  Layers (step eval)"):
        hook_name = layer_to_hook[layer_int]
        safe_name = hook_name.replace(".", "_")
        direction = vectors[layer_int]  # [d_model]

        # Load all shard tensors for this layer into a flat [T_total, d_model] tensor
        shard_parts = []
        for sid in range(num_shards):
            shard_path = raw_dir / safe_name / f"shard_{sid:04d}.pt"
            shard_parts.append(
                torch.load(shard_path, map_location="cpu", weights_only=False).float()
            )
        all_acts = torch.cat(shard_parts, dim=0)  # [T_total, d_model]
        del shard_parts

        # Iterate over (sample_idx, step_idx) groups
        for (sample_idx, step_idx), row_indices in step_groups.items():
            # Step mean activation: average token activations within this step
            step_acts = all_acts[row_indices]          # [T_step, d_model]
            step_mean = step_acts.mean(dim=0)          # [d_model]

            # Step label: majority vote of is_correct across its tokens
            # (tokens within a step all share the same label, but vote for robustness)
            correct_votes   = sum(1 for r in row_indices if all_meta[r].get("is_correct") is True)
            incorrect_votes = sum(1 for r in row_indices if all_meta[r].get("is_correct") is False)
            if correct_votes == 0 and incorrect_votes == 0:
                continue  # unlabelled step — skip
            step_is_correct = correct_votes >= incorrect_votes

            # Cosine similarity of this step's mean activation with the steering vector
            cos_sim = F.cosine_similarity(
                step_mean.unsqueeze(0), direction.unsqueeze(0), dim=-1
            ).item()

            label = "correct" if step_is_correct else "incorrect"
            step_scores[layer_int][label].append(cos_sim)

        del all_acts

    n_correct   = sum(len(step_scores[available_layers[0]]["correct"])   for _ in [0])
    n_incorrect = sum(len(step_scores[available_layers[0]]["incorrect"]) for _ in [0])
    print(f"  Step-level: {n_correct} correct steps, {n_incorrect} incorrect steps "
          f"(at layer {available_layers[0]}; same count for all layers)")

    return step_scores


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

    scores: dict[int, dict[str, list]] = {l: defaultdict(list) for l in available_layers}

    for label, path in zip(args.eval_labels, args.eval_datasets):
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
                    hs   = out.hidden_states[layer_int]
                    mask = inputs["attention_mask"]

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
# Helpers: compute metrics dict and print a summary table
# ---------------------------------------------------------------------------
def compute_metrics(scores: dict[int, dict[str, list]],
                    pos_label: str, neg_labels: list[str]) -> dict:
    """Return metrics[(layer, neg_label)] for all layers and neg labels."""
    metrics = {}
    for layer in available_layers:
        pos_scores = np.array(scores[layer][pos_label])
        for neg_label in neg_labels:
            neg_scores = np.array(scores[layer][neg_label])
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue
            metrics[(layer, neg_label)] = compute_discriminability(pos_scores, neg_scores)
    return metrics


def print_summary(metrics: dict, neg_labels: list[str], title: str) -> tuple:
    """Print a formatted summary table. Returns (best_auroc, best_layer)."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    best_auroc, best_layer = -1, None
    for neg_label in neg_labels:
        print(f"\n  vs '{neg_label}'")
        print(f"  {'Layer':>6} | {'AUROC':>8} | {'Cohen d':>8} | {'gap':>8} | {'mean+':>8} | {'mean-':>8} | {'n+':>6} | {'n-':>6}")
        print(f"  {'-'*76}")
        for layer in available_layers:
            m = metrics.get((layer, neg_label), {})
            auroc = m.get("auroc", 0.5)
            print(f"  {layer:>6} | {auroc:>8.4f} | {m.get('cohens_d',0):>+8.4f} | "
                  f"{m.get('gap',0):>+8.4f} | {m.get('mean_pos',0):>8.4f} | "
                  f"{m.get('mean_neg',0):>8.4f} | {m.get('n_pos',0):>6} | {m.get('n_neg',0):>6}")
            if auroc > best_auroc:
                best_auroc, best_layer = auroc, (layer, neg_label)
    print(f"\n{'='*90}")
    if best_layer:
        print(f"  ★  Best: layer {best_layer[0]} vs '{best_layer[1]}'  "
              f"AUROC={best_auroc:.4f}, Cohen's d={metrics[best_layer].get('cohens_d',0):+.4f}")
    print(f"{'='*90}\n")
    return best_auroc, best_layer


# ---------------------------------------------------------------------------
# Run eval
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
step_scores = None

if all_pt:
    print("Mode: pre-computed .pt activations — labels from is_fully_correct")
    print("\n[1/2] Sample-level evaluation ...")
    scores = eval_from_pt()
    pos_label  = "correct"
    neg_labels = ["incorrect"]

    if args.raw_activations_dir is not None:
        raw_dir = Path(args.raw_activations_dir)
        print(f"\n[2/2] Step-level evaluation from raw shards in {raw_dir} ...")
        step_scores = eval_steps_from_raw(raw_dir)
    else:
        print("\n[2/2] Skipping step-level evaluation (no --raw_activations_dir given)")
else:
    print("Mode: .jsonl raw text (model forward pass)")
    scores = eval_from_jsonl()
    pos_label = next(
        (l for l in args.eval_labels if "reason" in l.lower() and "non" not in l.lower()),
        args.eval_labels[0]
    )
    neg_labels = [l for l in args.eval_labels if l != pos_label]

print(f"\nPositive: '{pos_label}', Negatives: {neg_labels}")

# ---------------------------------------------------------------------------
# Metrics + summaries
# ---------------------------------------------------------------------------
sample_metrics = compute_metrics(scores, pos_label, neg_labels)
print_summary(sample_metrics, neg_labels, "SAMPLE-LEVEL COSINE EVALUATION SUMMARY")

step_metrics = {}
if step_scores is not None:
    step_metrics = compute_metrics(step_scores, "correct", ["incorrect"])
    print_summary(step_metrics, ["incorrect"], "STEP-LEVEL COSINE EVALUATION SUMMARY")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def safe_float(x):
    f = float(x)
    return None if (np.isnan(f) or np.isinf(f)) else f

def serialise_metrics(metrics: dict) -> dict:
    return {
        f"layer{layer}__vs_{neg_label}": {
            k: safe_float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }
        for (layer, neg_label), m in metrics.items()
    }

def serialise_raw_scores(scores: dict[int, dict[str, list]],
                          labels: list[str]) -> dict:
    out = {}
    for layer in available_layers:
        for label in labels:
            arr = scores[layer].get(label, [])
            if arr:
                out[f"layer{layer}__{label}"] = {
                    "cosine_mean": safe_float(np.mean(arr)),
                    "cosine_std":  safe_float(np.std(arr)),
                    "n":           len(arr),
                }
    return out

all_sample_labels = [pos_label] + neg_labels

out = {
    "sample_level": {
        "direction_metrics": serialise_metrics(sample_metrics),
        "raw_scores":        serialise_raw_scores(scores, all_sample_labels),
    },
}

if step_scores is not None:
    out["step_level"] = {
        "direction_metrics": serialise_metrics(step_metrics),
        "raw_scores":        serialise_raw_scores(step_scores, ["correct", "incorrect"]),
    }

results_path = os.path.join(args.output_dir, "eval_mean_diff_cosine.json")
with open(results_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved → {results_path}")