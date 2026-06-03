"""
Reasoning Vector Layer & Direction Selection Evaluation
========================================================

Systematic evaluation framework for choosing the optimal (vector_type, layer)
pair for a reasoning direction vector. 
The core metric you want is discriminability: 
how well does the vector separate reasoning from non-reasoning text, measured as a binary classification signal? 
Evaluates discriminability between
reasoning and non-reasoning text using:
  - Cohen's d (effect size)
  - AUROC (classification performance)
  - Selectivity ratio (activation specificity)
  - Statistical significance (Mann-Whitney U test)

unsure whether transofomers' AutoTokenizer applies the chat template by default!

control for:
- chat template
- keywords in format (Problem, Reasoning, Explanation, The correct answer is ...) (to do in build_eval_dataset_for_layer_selection.py)

Supports three vector sources via --vector_source:
  - "reasoning"  : mean-difference reasoning direction vectors from the .pt vector file
  - "lr"         : LR classifier weight vectors from lr_learned_weights.pt
  - "both"       : evaluate both side by side
"""

import os
import argparse
import json
import sys
import re

# ==========================================
# 1. Parse Arguments FIRST (before CUDA imports)
# ==========================================
parser = argparse.ArgumentParser(
    description="Evaluate reasoning vectors across layers for optimal selection."
)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--vector_file", type=str, default=None,
                    help="Path to .pt file with reasoning vectors (must have 'layers' key). "
                         "Required when --vector_source is 'reasoning' or 'both'.")
parser.add_argument("--lr_weights_file", type=str, default=None,
                    help="Path to lr_learned_weights.pt from the LR classifier. "
                         "Required when --vector_source is 'lr' or 'both'.")
parser.add_argument("--vector_source", type=str, default="both",
                    choices=["reasoning", "lr", "both"],
                    help="Which vector families to evaluate and plot: "
                         "'reasoning' = mean-difference directions only, "
                         "'lr' = LR classifier weights only, "
                         "'both' = all vectors side by side.")
parser.add_argument("--eval_datasets", type=str, nargs="+",
                    default=[
                        "/home/ines/Reasoning-activations/reasoning_datasets/eval_data_layer_selection/reasoning_eval.jsonl",
                        "/home/ines/Reasoning-activations/reasoning_datasets/eval_data_layer_selection/non_reasoning_hard.jsonl",
                        "/home/ines/Reasoning-activations/reasoning_datasets/eval_data_layer_selection/non_reasoning_easy.jsonl",
                    ],
                    help="Paths to evaluation .jsonl files (one per category)")
parser.add_argument("--eval_labels", type=str, nargs="+",
                    default=["reasoning", "non_reasoning_hard", "non_reasoning_easy"],
                    help="Labels for each dataset (e.g., 'reasoning' 'non_reasoning_hard' 'non_reasoning_easy')")

parser.add_argument("--target_layers", type=int, nargs="+", default=list(range(18, 29)))
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/02_validation_exp/layer_selection")
parser.add_argument("--plot_file", type=str, default=None,
                    help="Path for output plot (default: dynamically generated based on parameters)")
parser.add_argument("--token_positions", type=str, default="last",
                    choices=["last", "mean_all", "mean_prompt",
                             "topk_mean", "percentile_95", "percentile_99"],
                    help="Aggregation mode. 'topk_mean'/'percentile_*' compute per-token "
                         "cosine sims across the full sequence, then aggregate.")
parser.add_argument("--topk_pct", type=float, default=10.0,
                    help="Top-K%% of tokens to average over (only used with topk_mean)")
# Defaults to False. If you include --chat_template, it becomes True.
parser.add_argument("--chat_template", action="store_true", 
                    help="Include this flag to apply the chat template.")

args = parser.parse_args()

# --- Validate required files for selected vector_source ---
if args.vector_source in ("reasoning", "both") and args.vector_file is None:
    print("ERROR: --vector_file is required when --vector_source is 'reasoning' or 'both'.")
    sys.exit(1)
if args.vector_source in ("lr", "both") and args.lr_weights_file is None:
    print("ERROR: --lr_weights_file is required when --vector_source is 'lr' or 'both'.")
    sys.exit(1)

if len(args.eval_datasets) != len(args.eval_labels):
    print("ERROR: --eval_datasets and --eval_labels must have the same number of entries.")
    sys.exit(1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==========================================
# 2. Imports (after CUDA env is set)
# ==========================================
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# Candidate vectors to evaluate
# ==========================================
REASONING_VECTORS = [
    "reasoning_direction_token_cleaned",
    "reasoning_direction_sample_cleaned",
    "reasoning_direction_step_cleaned",
    "reasoning_direction_token",
    "reasoning_direction_sample",
    "reasoning_direction_step",
]

LR_WEIGHT_VECTORS = [
    "lr_weight_token",
    "lr_weight_step",
    "lr_weight_sample",
]

# ==========================================
# Plotting style registry
# ==========================================
# Colors: one hue per granularity. Reasoning directions = blue/orange/green.
# LR weights = pink/purple/brown. Solid = cleaned or LR, dashed = raw.
PLOT_COLORS = {
    "reasoning_direction_token_cleaned":  "#2196F3",
    "reasoning_direction_sample_cleaned": "#FF9800",
    "reasoning_direction_step_cleaned":   "#4CAF50",
    "reasoning_direction_token":          "#2196F3",
    "reasoning_direction_sample":         "#FF9800",
    "reasoning_direction_step":           "#4CAF50",
    "lr_weight_token":                    "#E91E63",
    "lr_weight_step":                     "#9C27B0",
    "lr_weight_sample":                   "#795548",
}

PLOT_LINESTYLES = {
    "reasoning_direction_token_cleaned":  "-",
    "reasoning_direction_sample_cleaned": "-",
    "reasoning_direction_step_cleaned":   "-",
    "reasoning_direction_token":          "--",
    "reasoning_direction_sample":         "--",
    "reasoning_direction_step":           "--",
    "lr_weight_token":                    "-",
    "lr_weight_step":                     "-",
    "lr_weight_sample":                   "-",
}

PLOT_SHORT_NAMES = {
    "reasoning_direction_token_cleaned":  "Token (cleaned)",
    "reasoning_direction_sample_cleaned": "Sample (cleaned)",
    "reasoning_direction_step_cleaned":   "Step (cleaned)",
    "reasoning_direction_token":          "Token (raw)",
    "reasoning_direction_sample":         "Sample (raw)",
    "reasoning_direction_step":           "Step (raw)",
    "lr_weight_token":                    "LR Token",
    "lr_weight_step":                     "LR Step",
    "lr_weight_sample":                   "LR Sample",
}


# ==========================================
# LR learned-weight loader
# ==========================================
def load_lr_learned_weights(
    weights_path: str,
    target_layers: list[int],
    device,
    dtype=torch.bfloat16,
) -> dict:
    """
    Load LR classifier weight vectors from lr_learned_weights.pt and reshape
    them into the same  {layer: {vector_name: tensor}}  structure used by the
    evaluation pipeline.

    The LR file stores:
        {str(layer): {"token": w, "step": w, "sample": w}}

    We emit:
        {int(layer): {"lr_weight_token": w, "lr_weight_step": w, "lr_weight_sample": w}}

    Vectors are L2-normalised so that downstream cosine similarity is meaningful
    (the LR weight magnitudes are arbitrary and depend on regularisation).
    """
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)

    vectors: dict[int, dict[str, torch.Tensor]] = {}
    for layer in target_layers:
        layer_key = str(layer)
        if layer_key not in raw:
            print(f"  Warning: layer {layer} not found in {weights_path}, skipping")
            continue
        vectors[layer] = {}
        for gran, w in raw[layer_key].items():          # gran ∈ {"token", "step", "sample"}
            vec = w.to(device=device, dtype=dtype)
            # vec = F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0)   # unit-norm for cosine sim
            vectors[layer][f"lr_weight_{gran}"] = vec

    loaded_summary = {l: list(v.keys()) for l, v in vectors.items()}
    print(f"  Loaded LR weight vectors: {loaded_summary}")
    return vectors


# ==========================================
# Data loading helpers
# ==========================================
def load_eval_data_with_chat_template(file_path: str, tokenizer, label: str) -> list[dict]:
    """Load a .jsonl evaluation file. Each line needs a 'problem' field."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Flexible: accept 'problem', 'text', 'prompt', or 'content' as the text field
            text = data.get("problem") or data.get("text") or data.get("prompt") or data.get("content", "")
            if not text:
                continue

            messages = [{"role": "user", "content": text}]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            samples.append({
                "id": data.get("problem_id", data.get("id", f"{label}_{len(samples)}")),
                "category": label,
                "prompt_str": prompt_str,
                "text_preview": text[:80],
            })
    return samples

def load_eval_data(file_path: str, tokenizer, label: str) -> list[dict]:
    """Load a .jsonl evaluation file. Each line needs a 'problem' field."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Flexible: accept 'problem', 'text', 'prompt', or 'content' as the text field
            text = data.get("problem") or data.get("text") or data.get("prompt") or data.get("content", "")
            if not text:
                continue

            # The raw text is passed directly. 
            # Your downstream loop (tokenizer(batch_texts, ...)) will handle the actual tokenization.
            samples.append({
                "id": data.get("problem_id", data.get("id", f"{label}_{len(samples)}")),
                "category": label,
                "prompt_str": text,
                "text_preview": text[:80],
            })
    return samples


# ==========================================
# Activation extraction helpers
# ==========================================
def extract_boundary_activation(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    mode: str = "last",
) -> torch.Tensor:
    """
    Extract activation vector(s) from a batch of hidden states.

    Args:
        hidden_states: output of model(..., output_hidden_states=True).hidden_states
        attention_mask: the attention mask from tokenizer
        layer: which layer index to extract from
        mode: 'last' (pre-generation boundary), 'mean_all', or 'mean_prompt'

    Returns:
        [batch_size, d_model] tensor
    """
    layer_hidden = hidden_states[layer]  # [B, seq_len, d_model]

    if mode == "last":
        # Last non-pad token (pre-generation boundary)
        # For left-padded inputs, this is simply [:, -1, :]
        return layer_hidden[:, -1, :]

    elif mode == "mean_all":
        # Mean over all non-pad tokens
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        return (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    elif mode == "mean_prompt":
        # Mean over non-pad tokens, excluding last 5 (template tokens)
        mask = attention_mask.clone()
        for b in range(mask.shape[0]):
            seq_len = mask[b].sum().item()
            if seq_len > 5:
                # Zero out last 5 real tokens
                real_end = mask[b].nonzero()[-1].item()
                mask[b, max(0, real_end - 4) : real_end + 1] = 0
        mask = mask.unsqueeze(-1).float()
        return (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def per_token_cosine_aggregate(
    hidden_states,
    attention_mask: torch.Tensor,
    layer: int,
    target_vec: torch.Tensor,
    mode: str = "topk_mean",
    topk_pct: float = 10.0,
) -> torch.Tensor:
    """
    Compute per-token cosine similarity with target_vec, then aggregate.

    Tracks the cosine similarity at every single token across a sequence, 
    isolates the top topk_pct% highest-scoring tokens, and averages them.

    Args:
        hidden_states: model output hidden_states tuple
        attention_mask: [B, seq_len]
        layer: layer index
        target_vec: [d_model] reasoning direction vector
        mode: 'topk_mean', 'percentile_95', 'percentile_99'
        topk_pct: percentage of tokens for topk_mean (default 10%)

    Returns:
        [B] tensor — one scalar score per sample
    """
    layer_hidden = hidden_states[layer]  # [B, seq_len, d_model]
    B, S, D = layer_hidden.shape

    # Per-token cosine sim: [B, seq_len]
    cos_sims = F.cosine_similarity(
        layer_hidden, target_vec.unsqueeze(0).unsqueeze(0).expand(B, S, -1), dim=-1
    )

    # Mask out pad tokens by setting them to -inf so they never enter top-k/percentile
    pad_mask = attention_mask == 0  # True where padded
    cos_sims = cos_sims.masked_fill(pad_mask, float("-inf"))

    # Number of real tokens per sample
    real_lengths = attention_mask.sum(dim=1)  # [B]

    if mode == "topk_mean":
        scores = torch.zeros(B, device=cos_sims.device)
        for b in range(B):
            real_sims = cos_sims[b][attention_mask[b].bool()]  # padding-side agnostic
            if len(real_sims) == 0:
                scores[b] = 0.0
                continue
            k = max(1, int(len(real_sims) * topk_pct / 100.0))
            topk_vals = real_sims.topk(k).values
            scores[b] = topk_vals.mean()
        return scores

    elif mode == "percentile_95":
        pct = 95.0
    elif mode == "percentile_99":
        pct = 99.0
    else:
        raise ValueError(f"Unknown per-token aggregation mode: {mode}")

    # Percentile path
    scores = torch.zeros(B, device=cos_sims.device)
    for b in range(B):
        real_sims = cos_sims[b][attention_mask[b].bool()]  # padding-side agnostic
        if len(real_sims) == 0:
            scores[b] = 0.0
            continue
        idx = min(int(len(real_sims) * pct / 100.0), len(real_sims) - 1)
        scores[b] = real_sims.sort().values[idx]
    return scores


# ==========================================
# Discriminability metrics
# ==========================================
def compute_discriminability_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> dict:
    """
    Compute discrimination metrics between positive (reasoning) and negative (non-reasoning) score distributions.

    Returns dict with:
        - cohens_d: standardized effect size (positive = reasoning > non-reasoning)
        - auroc: area under ROC curve (0.5 = chance, 1.0 = perfect separation)
        - selectivity_ratio: mean_pos / (|mean_neg| + eps), higher = more selective
        - mann_whitney_p: p-value for distribution separation
        - mean_pos, mean_neg, std_pos, std_neg: distribution statistics
        - gap: mean_pos - mean_neg (raw difference)
    """
    # Filter out any residual inf/nan values
    positive_scores = positive_scores[np.isfinite(positive_scores)]
    negative_scores = negative_scores[np.isfinite(negative_scores)]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return {
            "cohens_d": 0.0, "auroc": 0.5, "selectivity_ratio": 0.0,
            "mann_whitney_p": 1.0, "mean_pos": 0.0, "mean_neg": 0.0,
            "std_pos": 0.0, "std_neg": 0.0, "gap": 0.0,
            "n_pos": len(positive_scores), "n_neg": len(negative_scores),
        }

    mean_pos = np.mean(positive_scores)
    mean_neg = np.mean(negative_scores)
    std_pos = np.std(positive_scores, ddof=1)
    std_neg = np.std(negative_scores, ddof=1)

    # Cohen's d (pooled std)
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    pooled_std = np.sqrt(
        ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) / (n_pos + n_neg - 2)
    )
    cohens_d = (mean_pos - mean_neg) / (pooled_std + 1e-10)

    # AUROC
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([positive_scores, negative_scores])
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    # Selectivity ratio
    selectivity = mean_pos / (abs(mean_neg) + 1e-8)

    # Mann-Whitney U test
    try:
        _, p_value = stats.mannwhitneyu(positive_scores, negative_scores, alternative="greater")
    except ValueError:
        p_value = 1.0

    return {
        "cohens_d": cohens_d,
        "auroc": auroc,
        "selectivity_ratio": selectivity,
        "mann_whitney_p": p_value,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "std_pos": std_pos,
        "std_neg": std_neg,
        "gap": mean_pos - mean_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ==========================================
# Plotting
# ==========================================
def plot_layer_selection(
    all_metrics: dict,
    sorted_layers: list[int],
    neg_labels: list[str],
    save_path: str,
    subtitle: str = None,  
):
    """
    Create a multi-panel figure showing discriminability metrics across layers.

    Layout:
        Row 1: AUROC per vector type (one subplot per negative category)
        Row 2: Cohen's d per vector type
        Row 3: Mean cosine similarity (positive vs each negative)
    """
    n_neg = len(neg_labels)
    vec_names = list(all_metrics.keys())

    fig = plt.figure(figsize=(6 * n_neg, 14))
    gs = gridspec.GridSpec(3, n_neg, hspace=0.35, wspace=0.3)

    for col_idx, neg_label in enumerate(neg_labels):
        # --- Row 1: AUROC ---
        ax1 = fig.add_subplot(gs[0, col_idx])
        for v_name in vec_names:
            if v_name not in all_metrics:
                continue
            aurocs = [
                all_metrics[v_name].get((layer, neg_label), {}).get("auroc", 0.5)
                for layer in sorted_layers
            ]
            ax1.plot(
                sorted_layers, aurocs,
                marker="o", markersize=4,
                linestyle=PLOT_LINESTYLES.get(v_name, "-"),
                color=PLOT_COLORS.get(v_name, "gray"),
                linewidth=2,
                label=PLOT_SHORT_NAMES.get(v_name, v_name),
            )
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
        ax1.set_ylabel("AUROC", fontsize=11)
        ax1.set_title(f"vs {neg_label}", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.3, 1.0)
        ax1.set_xticks(sorted_layers)
        ax1.grid(True, alpha=0.3)
        if col_idx == 0:
            ax1.legend(fontsize=8, loc="best")

        # --- Row 2: Cohen's d ---
        ax2 = fig.add_subplot(gs[1, col_idx])
        for v_name in vec_names:
            if v_name not in all_metrics:
                continue
            ds = [
                all_metrics[v_name].get((layer, neg_label), {}).get("cohens_d", 0)
                for layer in sorted_layers
            ]
            ax2.plot(
                sorted_layers, ds,
                marker="s", markersize=4,
                linestyle=PLOT_LINESTYLES.get(v_name, "-"),
                color=PLOT_COLORS.get(v_name, "gray"),
                linewidth=2,
                label=PLOT_SHORT_NAMES.get(v_name, v_name),
            )
        ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Large effect (0.8)")
        ax2.set_ylabel("Cohen's d", fontsize=11)
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, alpha=0.3)

        # --- Row 3: Mean cosine sim distributions ---
        ax3 = fig.add_subplot(gs[2, col_idx])
        for v_name in vec_names:
            if v_name not in all_metrics:
                continue
            mean_pos = [
                all_metrics[v_name].get((layer, neg_label), {}).get("mean_pos", 0)
                for layer in sorted_layers
            ]
            mean_neg = [
                all_metrics[v_name].get((layer, neg_label), {}).get("mean_neg", 0)
                for layer in sorted_layers
            ]
            c = PLOT_COLORS.get(v_name, "gray")
            ls = PLOT_LINESTYLES.get(v_name, "-")
            ax3.plot(sorted_layers, mean_pos, marker="o", markersize=4,
                     linestyle=ls, color=c, linewidth=2, label=f"{PLOT_SHORT_NAMES.get(v_name)} (reasoning)")
            ax3.plot(sorted_layers, mean_neg, marker="x", markersize=4,
                     linestyle=ls, color=c, linewidth=1.5, alpha=0.5)

        ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax3.set_ylabel("Mean Cosine Sim", fontsize=11)
        ax3.set_xlabel("Layer", fontsize=11)
        ax3.set_xticks(sorted_layers)
        ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Reasoning Vector Layer Selection: Discriminability Metrics",
        fontsize=15, fontweight="bold", y=0.98,
    )
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=12, style='italic', color='dimgray')
        
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    print(f"Plot saved → {save_path}")


def print_recommendation(all_metrics: dict, sorted_layers: list[int], neg_labels: list[str]):
    """Print a summary table and recommend the best (vector, layer) pair."""
    print("\n" + "=" * 100)
    print("LAYER SELECTION SUMMARY")
    print("=" * 100)

    # For each vector, find best layer based on AUROC averaged over hard negatives
    # (prefer hard negatives if available, otherwise use all)
    hard_neg = [l for l in neg_labels if "hard" in l.lower()]
    eval_neg = hard_neg if hard_neg else neg_labels

    best_score = -1
    best_choice = None

    for v_name in all_metrics:
        print(f"\n{'─' * 80}")
        print(f"  Vector: {v_name}")
        print(f"{'─' * 80}")
        print(f"  {'Layer':>5} | {'AUROC':>8} | {'Cohen d':>8} | {'Gap':>8} | {'p-value':>10} | {'Mean(+)':>8} | {'Mean(-)':>8}")
        print(f"  {'─' * 75}")

        for layer in sorted_layers:
            # Average metrics over the evaluation negatives
            aurocs, ds, gaps, ps, mps, mns = [], [], [], [], [], []
            for neg_l in eval_neg:
                m = all_metrics[v_name].get((layer, neg_l), {})
                aurocs.append(m.get("auroc", 0.5))
                ds.append(m.get("cohens_d", 0))
                gaps.append(m.get("gap", 0))
                ps.append(m.get("mann_whitney_p", 1.0))
                mps.append(m.get("mean_pos", 0))
                mns.append(m.get("mean_neg", 0))

            avg_auroc = np.mean(aurocs)
            avg_d = np.mean(ds)
            avg_gap = np.mean(gaps)
            avg_p = np.mean(ps)
            avg_mp = np.mean(mps)
            avg_mn = np.mean(mns)

            marker = " ◀ BEST" if avg_auroc == max(
                np.mean([all_metrics[v_name].get((l, nl), {}).get("auroc", 0.5) for nl in eval_neg])
                for l in sorted_layers
            ) else ""

            print(f"  {layer:>5} | {avg_auroc:>8.4f} | {avg_d:>+8.4f} | {avg_gap:>+8.4f} | {avg_p:>10.2e} | {avg_mp:>+8.4f} | {avg_mn:>+8.4f}{marker}")

            if avg_auroc > best_score:
                best_score = avg_auroc
                best_choice = (v_name, layer, avg_auroc, avg_d)

    print(f"\n{'=' * 100}")
    if best_choice:
        v, l, auc, d = best_choice
        print(f"  ★ RECOMMENDATION: {v} at layer {l}")
        print(f"    AUROC = {auc:.4f}, Cohen's d = {d:+.4f}")
        print(f"    (evaluated against: {eval_neg})")
    print(f"{'=' * 100}\n")


# ==========================================
# Main
# ==========================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Reasoning Vector Layer Selection Evaluation ===")
    print(f"Model:          {args.model_name}")
    print(f"Vector source:  {args.vector_source}")
    print(f"Vector file:    {args.vector_file}")
    print(f"LR weights:     {args.lr_weights_file}")
    print(f"Datasets:       {list(zip(args.eval_labels, args.eval_datasets))}")
    print(f"Layers:         {args.target_layers}")
    print(f"Token mode:     {args.token_positions}")

    # ==========================================
    # Load model & tokenizer
    # ==========================================
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # ==========================================
    # Load vectors according to --vector_source
    # ==========================================
    reasoning_vectors: dict[int, dict[str, torch.Tensor]] = {}

    # --- Mean-difference reasoning direction vectors ---
    if args.vector_source in ("reasoning", "both"):
        print("Loading reasoning direction vectors...")
        vector_data = torch.load(args.vector_file, map_location="cpu", weights_only=False)
        layers_dict = vector_data["layers"]

        for layer in args.target_layers:
            layer_key = f"blocks.{layer}.hook_out"
            if layer_key not in layers_dict:
                print(f"  Warning: {layer_key} not found in vector_file, skipping layer {layer}")
                continue
            if layer not in reasoning_vectors:
                reasoning_vectors[layer] = {}
            for v_name in REASONING_VECTORS:
                if v_name in layers_dict[layer_key]:
                    vec = layers_dict[layer_key][v_name].to(
                        device=model.device, dtype=torch.bfloat16
                    )
                    reasoning_vectors[layer][v_name] = vec

        n_rv = sum(len(v) for v in reasoning_vectors.values())
        print(f"  Loaded {n_rv} reasoning direction entries across {len(reasoning_vectors)} layers")

    # --- LR classifier weight vectors ---
    if args.vector_source in ("lr", "both"):
        print("Loading LR classifier weight vectors...")
        lr_vecs = load_lr_learned_weights(
            args.lr_weights_file, args.target_layers, model.device
        )
        for layer, vdict in lr_vecs.items():
            if layer not in reasoning_vectors:
                reasoning_vectors[layer] = {}
            reasoning_vectors[layer].update(vdict)

    # --- Build availability sets ---
    available_layers = sorted(reasoning_vectors.keys())
    available_vectors = set()
    for layer in available_layers:
        available_vectors.update(reasoning_vectors[layer].keys())
    print(f"  Total: {len(available_vectors)} vector types across {len(available_layers)} layers")

    if not available_layers:
        print("ERROR: No vectors loaded for any target layer. Check your files.")
        sys.exit(1)

    # ==========================================
    # Load evaluation datasets
    # ==========================================
    print("Loading evaluation datasets...")
    all_samples = []
    category_counts = {}
    for label, fpath in zip(args.eval_labels, args.eval_datasets):
        
        # If --chat_template was included in the command, this is True
        if args.chat_template:
            samples = load_eval_data_with_chat_template(fpath, tokenizer, label)
        # If --chat_template was omitted, this is False
        else:
            samples = load_eval_data(fpath, tokenizer, label)
            
        all_samples.extend(samples)
        category_counts[label] = len(samples)
        print(f"  {label}: {len(samples)} samples (from {fpath})")

    # Identify which label is the positive (reasoning) class
    # Heuristic: label containing 'reason' or first label
    pos_label = None
    for label in args.eval_labels:
        if "reason" in label.lower() and "non" not in label.lower():
            pos_label = label
            break
    if pos_label is None:
        pos_label = args.eval_labels[0]
        print(f"  Warning: could not auto-detect positive label, using '{pos_label}'")
    neg_labels = [l for l in args.eval_labels if l != pos_label]
    print(f"  Positive class: '{pos_label}', Negative classes: {neg_labels}")

    # ==========================================
    # Forward pass & cosine similarity computation
    # ==========================================
    # Structure: results[v_name][layer][category] = list of cosine sims
    results = {
        v_name: {
            layer: {label: [] for label in args.eval_labels}
            for layer in available_layers
        }
        for v_name in available_vectors
    }

    is_per_token = args.token_positions in ("topk_mean", "percentile_95", "percentile_99")

    print(f"\nRunning inference ({len(all_samples)} samples, batch_size={args.batch_size})...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(all_samples), args.batch_size), desc="Batches"):
            batch = all_samples[i : i + args.batch_size]
            batch_texts = [s["prompt_str"] for s in batch]
            batch_cats = [s["category"] for s in batch]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=True)

            for layer in available_layers:
                if is_per_token:
                    # Per-token path: cosine sim at every token, then aggregate
                    for v_name in reasoning_vectors[layer]:
                        target_vec = reasoning_vectors[layer][v_name]
                        scores = per_token_cosine_aggregate(
                            outputs.hidden_states,
                            inputs["attention_mask"],
                            layer,
                            target_vec,
                            mode=args.token_positions,
                            topk_pct=args.topk_pct,
                        )
                        for j, cat in enumerate(batch_cats):
                            results[v_name][layer][cat].append(scores[j].item())
                else:
                    # Single-vector path: collapse to [B, d_model], then cosine sim
                    activations = extract_boundary_activation(
                        outputs.hidden_states,
                        inputs["attention_mask"],
                        layer,
                        mode=args.token_positions,
                    )
                    for v_name in reasoning_vectors[layer]:
                        target_vec = reasoning_vectors[layer][v_name].unsqueeze(0)
                        cos_sims = F.cosine_similarity(activations, target_vec, dim=-1)

                        for j, cat in enumerate(batch_cats):
                            results[v_name][layer][cat].append(cos_sims[j].item())

    # ==========================================
    # Compute discriminability metrics
    # ==========================================
    print("\nComputing discriminability metrics...")
    # all_metrics[v_name][(layer, neg_label)] = metrics_dict
    all_metrics = defaultdict(dict)

    for v_name in available_vectors:
        for layer in available_layers:
            pos_scores = np.array(results[v_name][layer][pos_label])
            for neg_label in neg_labels:
                neg_scores = np.array(results[v_name][layer][neg_label])
                if len(pos_scores) == 0 or len(neg_scores) == 0:
                    continue
                metrics = compute_discriminability_metrics(pos_scores, neg_scores)
                all_metrics[v_name][(layer, neg_label)] = metrics

    # ==========================================
    # Extract Dataset Information for file naming
    # ==========================================
    # Build a descriptive tag from whichever files were provided
    if args.vector_file:
        base_name = os.path.basename(args.vector_file)
        name_no_ext = os.path.splitext(base_name)[0]
        baseline_dataset = re.sub(r'^reasoning_vectors_(with_step_)?', '', name_no_ext)
    elif args.lr_weights_file:
        base_name = os.path.basename(args.lr_weights_file)
        name_no_ext = os.path.splitext(base_name)[0]
        baseline_dataset = name_no_ext
    else:
        baseline_dataset = "unknown"

    # Determine if chat template was used
    chat_suffix = "_with_chat_template" if args.chat_template else ""

    # Include vector_source in the file suffix so outputs don't overwrite each other
    source_tag = args.vector_source  # "reasoning", "lr", or "both"

    # Establish the conditional file suffix
    if args.token_positions == "topk_mean" and args.topk_pct is not None:
         file_suffix = f"{baseline_dataset}_{source_tag}_{args.token_positions}_{args.topk_pct}{chat_suffix}_not_normalized"
    else:
         file_suffix = f"{baseline_dataset}_{source_tag}_{args.token_positions}{chat_suffix}_not_normalized"

    # ==========================================
    # Save raw results
    # ==========================================
    # Convert to JSON-serializable format
    def safe_float(x):
        """Convert to JSON-safe float (replace inf/nan with None)."""
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return None
        return f

    json_results = {}
    for v_name in available_vectors:
        json_results[v_name] = {}
        for layer in available_layers:
            json_results[v_name][str(layer)] = {}
            for label in args.eval_labels:
                json_results[v_name][str(layer)][label] = {
                    "scores": [safe_float(s) for s in results[v_name][layer][label]],
                    "mean": safe_float(np.mean(results[v_name][layer][label])) if results[v_name][layer][label] else 0,
                    "std": safe_float(np.std(results[v_name][layer][label])) if results[v_name][layer][label] else 0,
                    "n": len(results[v_name][layer][label]),
                }
            for neg_label in neg_labels:
                key = f"metrics_vs_{neg_label}"
                m = all_metrics[v_name].get((layer, neg_label), {})
                json_results[v_name][str(layer)][key] = {
                    k: safe_float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in m.items()
                }

    results_path = os.path.join(args.output_dir, f"layer_selection_results_{file_suffix}.json")

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved → {results_path}")

    # ==========================================
    # Print summary & recommendation
    # ==========================================
    print_recommendation(all_metrics, available_layers, neg_labels)

    # ==========================================
    # Plot
    # ==========================================
    subtitle_text = f"Source: {source_tag} | Dataset: {baseline_dataset} | Token mode: {args.token_positions}"
    if args.token_positions == "topk_mean" and args.topk_pct is not None:
        subtitle_text += f" | Top-K: {args.topk_pct}%"

    plot_path = args.plot_file or os.path.join(args.output_dir, f"layer_selection_{file_suffix}.png")
    
    plot_layer_selection(all_metrics, available_layers, neg_labels, plot_path, subtitle=subtitle_text)

    print("Done!")


if __name__ == "__main__":
    main(args)

# ==========================================
# Example usage
# ==========================================

# --- Reasoning directions only ---
# python3 layer_selection_eval.py \
#   --vector_source reasoning \
#   --vector_file /path/to/reasoning_vectors.pt \
#   --eval_datasets .../reasoning_eval.jsonl .../non_reasoning_hard.jsonl .../non_reasoning_easy.jsonl \
#   --eval_labels reasoning non_reasoning_hard non_reasoning_easy \
#   --token_positions topk_mean --topk_pct 10

# --- LR classifier weights only ---
# python3 layer_selection_eval.py \
#   --vector_source lr \
#   --lr_weights_file /path/to/results/lr_classifier/lr_learned_weights.pt \
#   --eval_datasets .../reasoning_eval.jsonl .../non_reasoning_hard.jsonl .../non_reasoning_easy.jsonl \
#   --eval_labels reasoning non_reasoning_hard non_reasoning_easy \
#   --token_positions topk_mean --topk_pct 10

# --- Both side by side ---
# python3 layer_selection_eval.py \
#   --vector_source both \
#   --vector_file /path/to/reasoning_vectors.pt \
#   --lr_weights_file /path/to/results/lr_classifier/lr_learned_weights.pt \
#   --eval_datasets .../reasoning_eval.jsonl .../non_reasoning_hard.jsonl .../non_reasoning_easy.jsonl \
#   --eval_labels reasoning non_reasoning_hard non_reasoning_easy \
#   --token_positions topk_mean --topk_pct 10


# python3 layer_selection_eval.py --vector_source lr --lr_weights_file /home/ines/Reasoning-activations/results/lr_classifier/lr_learned_weights.pt  --token_positions topk_mean --topk_pct 10