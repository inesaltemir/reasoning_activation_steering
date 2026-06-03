"""
Evaluate LR-Learned Weight Vectors as Classifiers
===================================================

This script evaluates the logistic regression weight vectors from
`lr_learned_weights.pt` (produced by train_lr_classifier_streaming_extended.py)
on the evaluation datasets built by build_eval_dataset_for_layer_selection.py.

Two evaluation modes:
  1. DIRECTION MODE (cosine similarity) — treats each LR weight vector as a
     direction in activation space, computes cosine similarity with activations,
     and measures discriminability (Cohen's d, AUROC, selectivity ratio,
     Mann-Whitney p). Directly comparable to the reasoning_direction_* vectors
     evaluated in layer_selection_eval.py.

  2. CLASSIFIER MODE (dot product + intercept) — uses the full linear model
     score = w_raw · activation + intercept_input_space, and evaluates as a
     binary classifier (accuracy, F1, precision, recall, AUROC on the logit).

Usage:
  python3 eval_lr_vectors_as_classifiers.py 
 
"""

import os
import sys
import re
import argparse
import json
import warnings

# ==========================================
# 1. Parse arguments FIRST (before CUDA imports)
# ==========================================
parser = argparse.ArgumentParser(
    description="Evaluate LR classifier vectors on reasoning/non-reasoning eval data."
)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--lr_weights_file", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier/lr_learned_weights.pt",
                    help="Path to lr_learned_weights.pt (dict[layer][granularity] -> weight tensor)")
parser.add_argument("--lr_metrics_file", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier/lr_classifier_results.json",
                    help="Path to lr_classifier_results.json (for intercepts). "
                         "If not provided, only cosine-similarity (direction) evaluation is run.")
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
parser.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier/eval")
parser.add_argument("--token_positions", type=str, default="mean_all",
                    choices=["last", "mean_all", "mean_prompt",
                             "topk_mean", "percentile_95", "percentile_99"])
parser.add_argument("--topk_pct", type=float, default=10.0,
                    help="Top-K%% of tokens for topk_mean aggregation")
parser.add_argument("--chat_template", action="store_true",
                    help="Apply the chat template to evaluation prompts.")

args = parser.parse_args()

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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# Data loading
# ==========================================
def load_eval_data(file_path: str, tokenizer, label: str, apply_chat_template: bool = False):
    """Load a .jsonl evaluation file."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = (data.get("problem") or data.get("text")
                    or data.get("prompt") or data.get("content", ""))
            if not text:
                continue

            if apply_chat_template:
                messages = [{"role": "user", "content": text}]
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_str = text

            samples.append({
                "id": data.get("problem_id", data.get("id", f"{label}_{len(samples)}")),
                "category": label,
                "prompt_str": prompt_str,
                "text_preview": text[:80],
            })
    return samples


# ==========================================
# Activation extraction (from layer_selection_eval.py)
# ==========================================
def extract_boundary_activation(hidden_states, attention_mask, layer, mode="last"):
    """Extract activation vector(s) from hidden states."""
    layer_hidden = hidden_states[layer]

    if mode == "last":
        return layer_hidden[:, -1, :]
    elif mode == "mean_all":
        mask = attention_mask.unsqueeze(-1).float()
        return (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif mode == "mean_prompt":
        mask = attention_mask.clone()
        for b in range(mask.shape[0]):
            seq_len = mask[b].sum().item()
            if seq_len > 5:
                real_end = mask[b].nonzero()[-1].item()
                mask[b, max(0, real_end - 4):real_end + 1] = 0
        mask = mask.unsqueeze(-1).float()
        return (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def per_token_cosine_aggregate(hidden_states, attention_mask, layer,
                                target_vec, mode="topk_mean", topk_pct=10.0):
    """Per-token cosine similarity aggregated via top-k or percentile."""
    layer_hidden = hidden_states[layer]
    B, S, D = layer_hidden.shape

    cos_sims = F.cosine_similarity(
        layer_hidden, target_vec.unsqueeze(0).unsqueeze(0).expand(B, S, -1), dim=-1
    )
    pad_mask = attention_mask == 0
    cos_sims = cos_sims.masked_fill(pad_mask, float("-inf"))

    if mode == "topk_mean":
        scores = torch.zeros(B, device=cos_sims.device)
        for b in range(B):
            real_sims = cos_sims[b][attention_mask[b].bool()]
            if len(real_sims) == 0:
                continue
            k = max(1, int(len(real_sims) * topk_pct / 100.0))
            scores[b] = real_sims.topk(k).values.mean()
        return scores

    pct = 95.0 if mode == "percentile_95" else 99.0
    scores = torch.zeros(B, device=cos_sims.device)
    for b in range(B):
        real_sims = cos_sims[b][attention_mask[b].bool()]
        if len(real_sims) == 0:
            continue
        idx = min(int(len(real_sims) * pct / 100.0), len(real_sims) - 1)
        scores[b] = real_sims.sort().values[idx]
    return scores


def per_token_dot_aggregate(hidden_states, attention_mask, layer,
                             weight_vec, intercept, mode="topk_mean", topk_pct=10.0):
    """
    Per-token dot product (w · h + b) aggregated via top-k or percentile.
    This gives actual classifier scores rather than cosine similarity.
    """
    layer_hidden = hidden_states[layer]  # (B, S, D)
    B, S, D = layer_hidden.shape

    # dot product at each token: (B, S)
    dot_scores = (layer_hidden * weight_vec.unsqueeze(0).unsqueeze(0)).sum(dim=-1) + intercept
    pad_mask = attention_mask == 0
    dot_scores = dot_scores.masked_fill(pad_mask, float("-inf"))

    if mode == "topk_mean":
        scores = torch.zeros(B, device=dot_scores.device)
        for b in range(B):
            real = dot_scores[b][attention_mask[b].bool()]
            if len(real) == 0:
                continue
            k = max(1, int(len(real) * topk_pct / 100.0))
            scores[b] = real.topk(k).values.mean()
        return scores

    pct = 95.0 if mode == "percentile_95" else 99.0
    scores = torch.zeros(B, device=dot_scores.device)
    for b in range(B):
        real = dot_scores[b][attention_mask[b].bool()]
        if len(real) == 0:
            continue
        idx = min(int(len(real) * pct / 100.0), len(real) - 1)
        scores[b] = real.sort().values[idx]
    return scores


# ==========================================
# Discriminability metrics (from layer_selection_eval.py)
# ==========================================
def compute_discriminability_metrics(positive_scores, negative_scores):
    """Cohen's d, AUROC, selectivity ratio, Mann-Whitney p-value."""
    positive_scores = positive_scores[np.isfinite(positive_scores)]
    negative_scores = negative_scores[np.isfinite(negative_scores)]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return {
            "cohens_d": 0.0, "auroc": 0.5, "selectivity_ratio": 0.0,
            "mann_whitney_p": 1.0, "mean_pos": 0.0, "mean_neg": 0.0,
            "std_pos": 0.0, "std_neg": 0.0, "gap": 0.0,
            "n_pos": 0, "n_neg": 0,
        }

    mean_pos, mean_neg = np.mean(positive_scores), np.mean(negative_scores)
    std_pos = np.std(positive_scores, ddof=1)
    std_neg = np.std(negative_scores, ddof=1)
    n_pos, n_neg = len(positive_scores), len(negative_scores)

    pooled_std = np.sqrt(
        ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) / (n_pos + n_neg - 2)
    )
    cohens_d = (mean_pos - mean_neg) / (pooled_std + 1e-10)

    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([positive_scores, negative_scores])
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    selectivity = mean_pos / (abs(mean_neg) + 1e-8)

    try:
        _, p_value = stats.mannwhitneyu(positive_scores, negative_scores, alternative="greater")
    except ValueError:
        p_value = 1.0

    return {
        "cohens_d": cohens_d, "auroc": auroc, "selectivity_ratio": selectivity,
        "mann_whitney_p": p_value, "mean_pos": mean_pos, "mean_neg": mean_neg,
        "std_pos": std_pos, "std_neg": std_neg, "gap": mean_pos - mean_neg,
        "n_pos": n_pos, "n_neg": n_neg,
    }


def compute_classifier_metrics(pos_scores, neg_scores, threshold=0.0):
    """Binary classification metrics using score > threshold as positive."""
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    y_pred = (y_scores > threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(auroc),
        "threshold": threshold,
        "n_pos": len(pos_scores),
        "n_neg": len(neg_scores),
        "frac_predicted_pos": float(y_pred.mean()),
    }


# ==========================================
# Plotting
# ==========================================
def plot_results(direction_metrics, classifier_metrics, sorted_layers, neg_labels,
                 granularities, save_path, subtitle=None):
    """
    Multi-panel figure:
      Row 1: AUROC from direction (cosine) evaluation — one column per negative category
      Row 2: Cohen's d from direction evaluation
      Row 3: AUROC from classifier (dot+intercept) evaluation (if available)
    """
    n_neg = len(neg_labels)
    n_rows = 3 if classifier_metrics else 2

    colors = {"token": "#2196F3", "step": "#4CAF50", "sample": "#FF9800"}
    markers = {"token": "o", "step": "s", "sample": "D"}

    fig = plt.figure(figsize=(6 * n_neg, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_neg, hspace=0.35, wspace=0.3)

    for col_idx, neg_label in enumerate(neg_labels):
        # Row 1: Direction AUROC
        ax1 = fig.add_subplot(gs[0, col_idx])
        for gran in granularities:
            aurocs = [
                direction_metrics.get((gran, layer, neg_label), {}).get("auroc", 0.5)
                for layer in sorted_layers
            ]
            ax1.plot(sorted_layers, aurocs, marker=markers.get(gran, "o"),
                     markersize=5, color=colors.get(gran, "gray"),
                     linewidth=2, label=f"LR {gran}")
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.set_ylabel("AUROC (cosine direction)", fontsize=10)
        ax1.set_title(f"vs {neg_label}", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.3, 1.05)
        ax1.set_xticks(sorted_layers)
        ax1.grid(True, alpha=0.3)
        if col_idx == 0:
            ax1.legend(fontsize=9, loc="best")

        # Row 2: Cohen's d
        ax2 = fig.add_subplot(gs[1, col_idx])
        for gran in granularities:
            ds = [
                direction_metrics.get((gran, layer, neg_label), {}).get("cohens_d", 0)
                for layer in sorted_layers
            ]
            ax2.plot(sorted_layers, ds, marker=markers.get(gran, "s"),
                     markersize=5, color=colors.get(gran, "gray"),
                     linewidth=2, label=f"LR {gran}")
        ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Large effect")
        ax2.set_ylabel("Cohen's d", fontsize=10)
        ax2.set_xlabel("Layer", fontsize=10)
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, alpha=0.3)

        # Row 3: Classifier AUROC (dot + intercept)
        if classifier_metrics:
            ax3 = fig.add_subplot(gs[2, col_idx])
            for gran in granularities:
                aurocs = [
                    classifier_metrics.get((gran, layer, neg_label), {}).get("auroc", 0.5)
                    for layer in sorted_layers
                ]
                ax3.plot(sorted_layers, aurocs, marker=markers.get(gran, "o"),
                         markersize=5, color=colors.get(gran, "gray"),
                         linewidth=2, label=f"LR {gran}")
            ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            ax3.set_ylabel("AUROC (classifier)", fontsize=10)
            ax3.set_xlabel("Layer", fontsize=10)
            ax3.set_ylim(0.3, 1.05)
            ax3.set_xticks(sorted_layers)
            ax3.grid(True, alpha=0.3)

    fig.suptitle("LR Classifier Vectors: Layer Selection Evaluation",
                 fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=11,
                 style="italic", color="dimgray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved → {save_path}")


# ==========================================
# Summary & recommendation
# ==========================================
def print_recommendation(direction_metrics, classifier_metrics, sorted_layers,
                         neg_labels, granularities):
    hard_neg = [l for l in neg_labels if "hard" in l.lower()]
    eval_neg = hard_neg if hard_neg else neg_labels

    print("\n" + "=" * 110)
    print("  LR CLASSIFIER VECTOR EVALUATION SUMMARY")
    print("=" * 110)

    best_score = -1
    best_choice = None

    for gran in granularities:
        print(f"\n{'─' * 90}")
        print(f"  Granularity: {gran.upper()}")
        print(f"{'─' * 90}")

        header = f"  {'Layer':>5} | {'cos AUROC':>10} | {'cos d':>8} | {'cos gap':>8}"
        if classifier_metrics:
            header += f" | {'clf AUROC':>10} | {'clf Acc':>8} | {'clf F1':>8}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for layer in sorted_layers:
            aurocs_cos, ds_cos, gaps_cos = [], [], []
            aurocs_clf, accs_clf, f1s_clf = [], [], []

            for neg_l in eval_neg:
                m = direction_metrics.get((gran, layer, neg_l), {})
                aurocs_cos.append(m.get("auroc", 0.5))
                ds_cos.append(m.get("cohens_d", 0))
                gaps_cos.append(m.get("gap", 0))

                if classifier_metrics:
                    mc = classifier_metrics.get((gran, layer, neg_l), {})
                    aurocs_clf.append(mc.get("auroc", 0.5))
                    accs_clf.append(mc.get("accuracy", 0))
                    f1s_clf.append(mc.get("f1", 0))

            avg_auroc = np.mean(aurocs_cos)
            row = f"  {layer:>5} | {avg_auroc:>10.4f} | {np.mean(ds_cos):>+8.4f} | {np.mean(gaps_cos):>+8.4f}"
            if classifier_metrics:
                row += f" | {np.mean(aurocs_clf):>10.4f} | {np.mean(accs_clf):>8.4f} | {np.mean(f1s_clf):>8.4f}"
            print(row)

            if avg_auroc > best_score:
                best_score = avg_auroc
                best_choice = (gran, layer, avg_auroc, np.mean(ds_cos))

    print(f"\n{'=' * 110}")
    if best_choice:
        g, l, auc, d = best_choice
        print(f"  ★ RECOMMENDATION: LR {g} weight vector at layer {l}")
        print(f"    Cosine AUROC = {auc:.4f}, Cohen's d = {d:+.4f}")
        print(f"    (evaluated against: {eval_neg})")
    print(f"{'=' * 110}\n")


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== LR Classifier Vector Evaluation ===")
    print(f"Model:           {args.model_name}")
    print(f"LR weights:      {args.lr_weights_file}")
    print(f"LR metrics:      {args.lr_metrics_file or '(not provided — classifier mode disabled)'}")
    print(f"Eval datasets:   {list(zip(args.eval_labels, args.eval_datasets))}")
    print(f"Layers:          {args.target_layers}")
    print(f"Token mode:      {args.token_positions}")

    # ==========================================
    # Load LR weight vectors
    # ==========================================
    print("\nLoading LR weight vectors...")
    lr_weights = torch.load(args.lr_weights_file, map_location="cpu", weights_only=False)
    # Structure: {layer_str: {granularity: tensor(d_model)}}

    # Load intercepts from the metrics JSON (if provided)
    intercepts = {}  # (layer_str, gran) -> float
    if args.lr_metrics_file and os.path.exists(args.lr_metrics_file):
        with open(args.lr_metrics_file, "r") as f:
            lr_metrics = json.load(f)
        for layer_str, layer_data in lr_metrics.items():
            if layer_str == "cross_weight_similarities":
                continue
            if isinstance(layer_data, dict):
                for gran, gran_data in layer_data.items():
                    if isinstance(gran_data, dict) and "intercept_input_space" in gran_data:
                        intercepts[(layer_str, gran)] = gran_data["intercept_input_space"]
        print(f"  Loaded {len(intercepts)} intercept values")
    else:
        print("  No metrics file → classifier mode disabled (direction-only evaluation)")

    # Index available (layer, granularity) pairs
    available_grans = set()
    lr_vectors = {}  # (layer_int, gran) -> tensor
    for layer_str, gran_dict in lr_weights.items():
        layer_int = int(layer_str)
        if layer_int not in args.target_layers:
            continue
        for gran, vec in gran_dict.items():
            lr_vectors[(layer_int, gran)] = vec
            available_grans.add(gran)

    granularities = sorted(available_grans)
    available_layers = sorted(set(l for l, _ in lr_vectors.keys()))
    print(f"  Available: {len(lr_vectors)} vectors — "
          f"granularities={granularities}, layers={available_layers}")

    # ==========================================
    # Load model & tokenizer
    # ==========================================
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = model.device

    # Move LR vectors to model device
    for key in lr_vectors:
        lr_vectors[key] = lr_vectors[key].to(device=device, dtype=torch.bfloat16)

    # ==========================================
    # Load evaluation datasets
    # ==========================================
    print("Loading evaluation datasets...")
    all_samples = []
    for label, fpath in zip(args.eval_labels, args.eval_datasets):
        samples = load_eval_data(fpath, tokenizer, label,
                                  apply_chat_template=args.chat_template)
        all_samples.extend(samples)
        print(f"  {label}: {len(samples)} samples")

    # Identify positive / negative labels
    pos_label = None
    for label in args.eval_labels:
        if "reason" in label.lower() and "non" not in label.lower():
            pos_label = label
            break
    if pos_label is None:
        pos_label = args.eval_labels[0]
    neg_labels = [l for l in args.eval_labels if l != pos_label]
    print(f"  Positive: '{pos_label}', Negative: {neg_labels}")

    # ==========================================
    # Forward pass & score computation
    # ==========================================
    # cosine_scores[(gran, layer)][category] = list of floats
    # classifier_scores[(gran, layer)][category] = list of floats
    cosine_scores = {
        (g, l): {label: [] for label in args.eval_labels}
        for g in granularities for l in available_layers
    }
    classifier_scores = {
        (g, l): {label: [] for label in args.eval_labels}
        for g in granularities for l in available_layers
    }

    is_per_token = args.token_positions in ("topk_mean", "percentile_95", "percentile_99")
    has_intercepts = len(intercepts) > 0

    print(f"\nRunning inference ({len(all_samples)} samples, batch_size={args.batch_size})...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(all_samples), args.batch_size), desc="Batches"):
            batch = all_samples[i:i + args.batch_size]
            batch_texts = [s["prompt_str"] for s in batch]
            batch_cats = [s["category"] for s in batch]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)

            for layer in available_layers:
                for gran in granularities:
                    key = (layer, gran)
                    if key not in lr_vectors:
                        continue

                    w_vec = lr_vectors[key]
                    intercept_val = intercepts.get((str(layer), gran), None)

                    if is_per_token:
                        # Cosine similarity path
                        cos_vals = per_token_cosine_aggregate(
                            outputs.hidden_states, inputs["attention_mask"],
                            layer, w_vec,
                            mode=args.token_positions, topk_pct=args.topk_pct,
                        )
                        for j, cat in enumerate(batch_cats):
                            cosine_scores[(gran, layer)][cat].append(cos_vals[j].item())

                        # Classifier (dot + intercept) path
                        if intercept_val is not None:
                            dot_vals = per_token_dot_aggregate(
                                outputs.hidden_states, inputs["attention_mask"],
                                layer, w_vec, intercept_val,
                                mode=args.token_positions, topk_pct=args.topk_pct,
                            )
                            for j, cat in enumerate(batch_cats):
                                classifier_scores[(gran, layer)][cat].append(dot_vals[j].item())
                    else:
                        # Single-vector path
                        activations = extract_boundary_activation(
                            outputs.hidden_states, inputs["attention_mask"],
                            layer, mode=args.token_positions,
                        )
                        # Cosine similarity
                        cos_sims = F.cosine_similarity(
                            activations, w_vec.unsqueeze(0), dim=-1
                        )
                        for j, cat in enumerate(batch_cats):
                            cosine_scores[(gran, layer)][cat].append(cos_sims[j].item())

                        # Classifier score: w · h + b
                        if intercept_val is not None:
                            dot_scores = (activations * w_vec.unsqueeze(0)).sum(dim=-1) + intercept_val
                            for j, cat in enumerate(batch_cats):
                                classifier_scores[(gran, layer)][cat].append(dot_scores[j].item())

    # ==========================================
    # Compute metrics
    # ==========================================
    print("\nComputing metrics...")

    # Direction (cosine) metrics
    direction_metrics = {}  # (gran, layer, neg_label) -> metrics dict
    for gran in granularities:
        for layer in available_layers:
            pos_scores = np.array(cosine_scores[(gran, layer)][pos_label])
            for neg_label in neg_labels:
                neg_scores_arr = np.array(cosine_scores[(gran, layer)][neg_label])
                if len(pos_scores) == 0 or len(neg_scores_arr) == 0:
                    continue
                m = compute_discriminability_metrics(pos_scores, neg_scores_arr)
                direction_metrics[(gran, layer, neg_label)] = m

    # Classifier metrics (if intercepts available)
    clf_metrics = {}
    if has_intercepts:
        for gran in granularities:
            for layer in available_layers:
                pos_scores = np.array(classifier_scores[(gran, layer)].get(pos_label, []))
                for neg_label in neg_labels:
                    neg_scores_arr = np.array(classifier_scores[(gran, layer)].get(neg_label, []))
                    if len(pos_scores) == 0 or len(neg_scores_arr) == 0:
                        continue
                    m = compute_classifier_metrics(pos_scores, neg_scores_arr, threshold=0.0)
                    clf_metrics[(gran, layer, neg_label)] = m

    # ==========================================
    # Save results
    # ==========================================
    chat_suffix = "_chat" if args.chat_template else ""
    if args.token_positions == "topk_mean":
        mode_suffix = f"{args.token_positions}_{args.topk_pct}"
    else:
        mode_suffix = args.token_positions
    file_tag = f"classifier_lr_vectors_{mode_suffix}{chat_suffix}"

    def safe_float(x):
        f = float(x)
        return None if (np.isnan(f) or np.isinf(f)) else f

    json_out = {"direction_metrics": {}, "classifier_metrics": {}, "raw_scores": {}}

    for (gran, layer, neg_l), m in direction_metrics.items():
        key = f"{gran}__layer{layer}__vs_{neg_l}"
        json_out["direction_metrics"][key] = {
            k: safe_float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }

    for (gran, layer, neg_l), m in clf_metrics.items():
        key = f"{gran}__layer{layer}__vs_{neg_l}"
        json_out["classifier_metrics"][key] = {
            k: safe_float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }

    # Save per-category score summaries
    for gran in granularities:
        for layer in available_layers:
            for label in args.eval_labels:
                skey = f"{gran}__layer{layer}__{label}"
                cos_arr = cosine_scores[(gran, layer)][label]
                entry = {
                    "cosine_mean": safe_float(np.mean(cos_arr)) if cos_arr else None,
                    "cosine_std": safe_float(np.std(cos_arr)) if cos_arr else None,
                    "n": len(cos_arr),
                }
                clf_arr = classifier_scores[(gran, layer)].get(label, [])
                if clf_arr:
                    entry["classifier_mean"] = safe_float(np.mean(clf_arr))
                    entry["classifier_std"] = safe_float(np.std(clf_arr))
                json_out["raw_scores"][skey] = entry

    results_path = os.path.join(args.output_dir, f"eval_results_{file_tag}.json")
    with open(results_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Results saved → {results_path}")

    # ==========================================
    # Print summary
    # ==========================================
    print_recommendation(direction_metrics, clf_metrics if has_intercepts else None,
                         available_layers, neg_labels, granularities)

    # ==========================================
    # Plot
    # ==========================================
    plot_path = os.path.join(args.output_dir, f"eval_plot_{file_tag}.png")
    subtitle = f"Token mode: {args.token_positions}"
    if args.token_positions == "topk_mean":
        subtitle += f" | Top-K: {args.topk_pct}%"
    plot_results(direction_metrics, clf_metrics if has_intercepts else None,
                 available_layers, neg_labels, granularities, plot_path, subtitle)

    print("Done!")


if __name__ == "__main__":
    main()