"""
Error Boundary Evaluation — Correct vs Incorrect Reasoning
============================================================

Tests whether reasoning direction vectors AND LR classifier models carry a
genuine reasoning-quality signal by checking discriminability at the *error
boundary* in ProcessBench samples.

Two scoring families:
  1. COSINE SIMILARITY — treats each vector (reasoning direction or LR weight)
     as a direction.  Score = cos(activation, vector).
  2. LR CLASSIFIER SCORE — uses the full trained linear model:
     score = w_raw · activation + intercept_input_space.
     Only available for LR weight vectors (requires intercepts from
     lr_classifier_results.json).

Three extraction modes at the boundary:
  • exact          — single token at the boundary
  • window_mean    — mean of ±W tokens around the boundary → score
  • window_max     — per-token score in ±W window, take max

NO MODEL LOADING, NO GPU, NO FORWARD PASSES.
All activations are read from disk (raw_activations/).

tests whether reasoning-direction vectors and the LR (logistic regression) classifier weights 
actually capture a "reasoning quality" signal, 
by checking how well they discriminate correct vs. incorrect reasoning 
right at the point where an error occurs in ProcessBench samples.

1. Build a boundary map — for each sample, find the token position where reasoning goes wrong 
(first is_correct=False step from ProcessBench labels). For correct samples (no error), 
it assigns a matched "control" position at the median relative error position across the dataset, 
so correct/incorrect comparisons are positionally controlled.

2. Extract activations around that boundary — 
reads pre-computed activations from disk (no model/GPU needed) for a window of ±window_size tokens, 
for each target layer.

3. Score each sample two ways:
Cosine similarity between the activation and a reasoning direction vector (works for any vector type).
LR classifier score (w·h + b) using the actual trained logistic regression weights + intercept 

Each is computed under three extraction modes: 
exact (boundary token only)
window_mean (average over the window)
window_max (best single token in the window).

Compute discriminability metrics comparing correct vs. incorrect scores
"""

import os
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# Arguments
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate reasoning vectors at the error boundary "
                    "(correct vs incorrect reasoning, no GPU needed)."
    )
    p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations",
                   help="Path to raw_activations/ directory produced by "
                        "run_fw_pass_with_step_averaging_storage.py")
    p.add_argument("--vector_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt",
                   help="Path to .pt reasoning vectors file "
                        "(must have 'layers' key with reasoning_direction_* vectors)")
    p.add_argument("--dataset_file", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/ProcessBench/dataset.jsonl",
                   help="Path to the original ProcessBench dataset.jsonl "
                        "(needed for per-sample 'label' field)")
    p.add_argument("--lr_weights_file", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier/lr_learned_weights.pt",
                   help="Optional: path to lr_learned_weights.pt for comparison")
    p.add_argument("--lr_metrics_file", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier/lr_classifier_results.json",
                   help="Path to lr_classifier_results.json (for intercepts). "
                        "Required for LR classifier scoring mode.")
    
    p.add_argument("--target_layers", type=int, nargs="+",
                   default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--window_size", type=int, default=5,
                   help="Half-width of the token window around the error boundary "
                        "(default 5 → 11-token window)")
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/error_boundary_eval_v2")
    return p.parse_args()


# ==========================================
# Constants
# ==========================================
REASONING_VECTOR_NAMES = [
    "reasoning_direction_token_cleaned",
    "reasoning_direction_sample_cleaned",
    "reasoning_direction_step_cleaned",
    "reasoning_direction_token",
    "reasoning_direction_sample",
    "reasoning_direction_step",
]

LR_GRANULARITIES = ["token", "step", "sample"]

EXTRACTION_MODES = ["exact", "window_mean", "window_max"]

# Two scoring methods
SCORING_COSINE = "cosine"
SCORING_LR = "lr_score"  # w · h + b


# ==========================================
# Metrics
# ==========================================
def compute_discriminability_metrics(positive_scores, negative_scores):
    """
    Discriminability metrics for continuous scores.
    Positive = correct samples, Negative = incorrect samples.
    """
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
        _, p_value = stats.mannwhitneyu(
            positive_scores, negative_scores, alternative="greater"
        )
    except ValueError:
        p_value = 1.0

    return {
        "cohens_d": cohens_d, "auroc": auroc,
        "selectivity_ratio": selectivity, "mann_whitney_p": p_value,
        "mean_pos": mean_pos, "mean_neg": mean_neg,
        "std_pos": std_pos, "std_neg": std_neg,
        "gap": mean_pos - mean_neg,
        "n_pos": n_pos, "n_neg": n_neg,
    }


def compute_classifier_metrics(positive_scores, negative_scores, threshold=0.0):
    """
    Binary classifier metrics using score > threshold as the decision rule.
    For the LR classifier, threshold=0.0 corresponds to the learned decision
    boundary (since sigmoid(0) = 0.5).

    positive_scores = scores for correct samples (should be > 0 if classifier
                      was trained with correct=1).
    negative_scores = scores for incorrect samples (should be < 0).
    """
    positive_scores = positive_scores[np.isfinite(positive_scores)]
    negative_scores = negative_scores[np.isfinite(negative_scores)]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return {
            "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "auroc": 0.5, "threshold": threshold,
            "n_pos": 0, "n_neg": 0,
            "frac_predicted_pos": 0.0,
        }

    y_true = np.concatenate([np.ones(len(positive_scores)),
                              np.zeros(len(negative_scores))])
    y_scores = np.concatenate([positive_scores, negative_scores])
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
        "n_pos": len(positive_scores),
        "n_neg": len(negative_scores),
        "frac_predicted_pos": float(y_pred.mean()),
    }


# ==========================================
# Phase 1: Build per-sample boundary map
# ==========================================
def build_boundary_map(raw_dir, dataset_file):
    """
    Build a map from sample_idx to boundary position.
    For incorrect samples: first token where is_correct = False.
    For correct samples: positional control at the median relative error position.
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    sample_index = index["sample_index"]
    num_shards = index["num_shards"]

    # Load ProcessBench labels
    dataset_labels, dataset_correct = {}, {}
    with open(dataset_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())
            dataset_labels[i] = sample.get("label", -1)
            dataset_correct[i] = sample.get("final_answer_correct", None)

    # Load all metadata
    print("  Loading token metadata from all shards...")
    all_meta = []
    shard_sizes = []
    for shard_id in range(num_shards):
        meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False)
        shard_sizes.append(len(meta))
        all_meta.extend(meta)
        if (shard_id + 1) % 5 == 0 or shard_id == num_shards - 1:
            print(f"    Loaded {shard_id + 1}/{num_shards} meta shards "
                  f"({len(all_meta)} total token rows)")

    # Build boundary map
    print("  Building boundary map...")
    boundary_map = {}

    for start_row, num_tokens, sample_idx in sample_index:
        sample_meta = all_meta[start_row: start_row + num_tokens]
        pb_label = dataset_labels.get(sample_idx, -1)
        pb_correct = dataset_correct.get(sample_idx, None)
        is_correct = (pb_label == -1) and (pb_correct is True)

        boundary_local_pos = None
        if not is_correct:
            for pos, m in enumerate(sample_meta):
                if m.get("is_correct") is False:
                    boundary_local_pos = pos
                    break

        relative_pos = None
        if boundary_local_pos is not None and num_tokens > 0:
            relative_pos = boundary_local_pos / num_tokens

        boundary_map[sample_idx] = {
            "is_correct": is_correct,
            "boundary_local_pos": boundary_local_pos,
            "total_tokens": num_tokens,
            "relative_pos": relative_pos,
            "global_start_row": start_row,
            "label": pb_label,
        }

    # Summary
    n_correct = sum(1 for v in boundary_map.values() if v["is_correct"])
    n_incorrect = sum(1 for v in boundary_map.values() if not v["is_correct"])
    n_with_boundary = sum(1 for v in boundary_map.values()
                          if v["boundary_local_pos"] is not None)
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}, "
          f"With boundary: {n_with_boundary}")

    # Median relative position for controls
    rel_positions = [v["relative_pos"] for v in boundary_map.values()
                     if v["relative_pos"] is not None]
    median_rel_pos = float(np.median(rel_positions)) if rel_positions else 0.5
    print(f"  Median error relative position: {median_rel_pos:.3f}")

    # Assign control positions to correct samples
    for info in boundary_map.values():
        if info["is_correct"]:
            control_pos = int(median_rel_pos * info["total_tokens"])
            control_pos = max(0, min(control_pos, info["total_tokens"] - 1))
            info["boundary_local_pos"] = control_pos
            info["relative_pos"] = control_pos / max(info["total_tokens"], 1)

    return boundary_map, shard_sizes, median_rel_pos


# ==========================================
# Phase 2: Extract boundary activations
# ==========================================
def extract_boundary_activations(raw_dir, hook_name, boundary_map,
                                  shard_sizes, window_size):
    """
    Load activations from shards and extract the boundary window
    for every sample in boundary_map.
    """
    raw_dir = Path(raw_dir)
    safe_name = hook_name.replace(".", "_")
    num_shards = len(shard_sizes)

    shard_boundaries = np.zeros(num_shards + 1, dtype=np.int64)
    for i, sz in enumerate(shard_sizes):
        shard_boundaries[i + 1] = shard_boundaries[i] + sz

    # Identify all needed global rows
    needed_rows = {}
    sample_window_info = {}

    for sample_idx, info in boundary_map.items():
        bpos = info["boundary_local_pos"]
        if bpos is None:
            continue
        total = info["total_tokens"]
        start = info["global_start_row"]

        win_start = max(0, bpos - window_size)
        win_end = min(total - 1, bpos + window_size)

        window_global_rows = []
        for local_pos in range(win_start, win_end + 1):
            grow = start + local_pos
            window_global_rows.append(grow)
            if grow not in needed_rows:
                needed_rows[grow] = []
            needed_rows[grow].append(sample_idx)

        sample_window_info[sample_idx] = {
            "global_rows": window_global_rows,
            "boundary_offset_in_window": bpos - win_start,
        }

    # Group rows by shard
    rows_by_shard = defaultdict(list)
    for grow in needed_rows:
        shard_id = int(np.searchsorted(shard_boundaries[1:], grow, side="right"))
        local_row = grow - int(shard_boundaries[shard_id])
        rows_by_shard[shard_id].append((grow, local_row))

    # Load activations
    row_to_activation = {}
    for shard_id in sorted(rows_by_shard.keys()):
        shard_path = raw_dir / safe_name / f"shard_{shard_id:04d}.pt"
        acts = torch.load(shard_path, weights_only=False).to(torch.float32)
        for grow, local_row in rows_by_shard[shard_id]:
            row_to_activation[grow] = acts[local_row]
        del acts

    # Assemble per-sample
    sample_activations = {}
    for sample_idx, win_info in sample_window_info.items():
        global_rows = win_info["global_rows"]
        boundary_idx = win_info["boundary_offset_in_window"]
        window_acts = torch.stack([row_to_activation[gr] for gr in global_rows])
        sample_activations[sample_idx] = {
            "exact": window_acts[boundary_idx],
            "window": window_acts,
            "boundary_idx_in_window": boundary_idx,
        }

    return sample_activations


# ==========================================
# Phase 3: Compute scores — BOTH cosine and LR
# ==========================================
def compute_all_scores(sample_activations, direction_vec, boundary_map,
                       intercept=None):
    """
    For each sample, compute scores in all extraction modes using:
      - cosine similarity (always)
      - LR classifier score w·h+b (only if intercept is provided)

    Returns
    -------
    results : dict
        {
            (scoring_method, extraction_mode): {
                "correct": [float, ...],
                "incorrect": [float, ...],
            },
            ...
        }
    """
    scoring_methods = [SCORING_COSINE]
    if intercept is not None:
        scoring_methods.append(SCORING_LR)

    results = {
        (sm, mode): {"correct": [], "incorrect": []}
        for sm in scoring_methods
        for mode in EXTRACTION_MODES
    }

    dvec = direction_vec.to(torch.float32)
    dvec_2d = dvec.unsqueeze(0)

    for sample_idx, acts in sample_activations.items():
        info = boundary_map[sample_idx]
        label_key = "correct" if info["is_correct"] else "incorrect"

        exact_act = acts["exact"]          # (d_model,)
        window_acts = acts["window"]       # (W, d_model)
        window_mean_act = window_acts.mean(dim=0)  # (d_model,)

        # ------ COSINE SIMILARITY ------
        # exact
        cos_exact = F.cosine_similarity(
            exact_act.unsqueeze(0), dvec_2d, dim=-1
        ).item()
        results[(SCORING_COSINE, "exact")][label_key].append(cos_exact)

        # window_mean
        cos_wmean = F.cosine_similarity(
            window_mean_act.unsqueeze(0), dvec_2d, dim=-1
        ).item()
        results[(SCORING_COSINE, "window_mean")][label_key].append(cos_wmean)

        # window_max (max cosine sim over tokens in window)
        per_token_cos = F.cosine_similarity(
            window_acts, dvec_2d.expand(window_acts.shape[0], -1), dim=-1
        )
        results[(SCORING_COSINE, "window_max")][label_key].append(
            per_token_cos.max().item()
        )

        # ------ LR CLASSIFIER SCORE (w · h + b) ------
        if intercept is not None:
            b = intercept

            # exact
            lr_exact = (exact_act * dvec).sum().item() + b
            results[(SCORING_LR, "exact")][label_key].append(lr_exact)

            # window_mean
            lr_wmean = (window_mean_act * dvec).sum().item() + b
            results[(SCORING_LR, "window_mean")][label_key].append(lr_wmean)

            # window_max (max dot-product score over tokens in window)
            per_token_dot = (window_acts * dvec.unsqueeze(0)).sum(dim=-1) + b
            results[(SCORING_LR, "window_max")][label_key].append(
                per_token_dot.max().item()
            )

    return results


# ==========================================
# Plotting
# ==========================================
def plot_results(all_metrics, sorted_layers, vec_names, save_path,
                 subtitle=None, scoring_method="cosine"):
    """
    3-row × 3-col figure for one scoring method.
    Columns = extraction modes.
    Rows = AUROC, Cohen's d, Mean score.
    """
    n_modes = len(EXTRACTION_MODES)
    palette = plt.cm.tab10.colors
    vec_colors = {vn: palette[i % len(palette)] for i, vn in enumerate(vec_names)}

    fig = plt.figure(figsize=(6 * n_modes, 14))
    gs = gridspec.GridSpec(3, n_modes, hspace=0.35, wspace=0.3)

    metric_key = "auroc"  # used for both discrim and classifier

    for col_idx, mode in enumerate(EXTRACTION_MODES):
        # Row 1: AUROC
        ax1 = fig.add_subplot(gs[0, col_idx])
        for vn in vec_names:
            vals = [
                all_metrics.get((vn, layer, scoring_method, mode), {}).get(metric_key, 0.5)
                for layer in sorted_layers
            ]
            short = vn.replace("reasoning_direction_", "rd:").replace("lr_weight_", "lr:")
            ax1.plot(sorted_layers, vals, marker="o", markersize=4,
                     color=vec_colors[vn], linewidth=2, label=short)
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.set_ylabel("AUROC", fontsize=10)
        ax1.set_title(f"Mode: {mode}", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.3, 1.0)
        ax1.set_xticks(sorted_layers)
        ax1.grid(True, alpha=0.3)
        if col_idx == 0:
            ax1.legend(fontsize=5, loc="best")

        # Row 2: Cohen's d (only for discriminability metrics)
        ax2 = fig.add_subplot(gs[1, col_idx])
        for vn in vec_names:
            ds = [
                all_metrics.get((vn, layer, scoring_method, mode), {}).get("cohens_d", 0)
                for layer in sorted_layers
            ]
            ax2.plot(sorted_layers, ds, marker="s", markersize=4,
                     color=vec_colors[vn], linewidth=2)
        ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.3)
        ax2.set_ylabel("Cohen's d", fontsize=10)
        ax2.set_xlabel("Layer", fontsize=10)
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, alpha=0.3)

        # Row 3: Mean scores (positive vs negative)
        ax3 = fig.add_subplot(gs[2, col_idx])
        for vn in vec_names:
            mp = [
                all_metrics.get((vn, layer, scoring_method, mode), {}).get("mean_pos", 0)
                for layer in sorted_layers
            ]
            mn = [
                all_metrics.get((vn, layer, scoring_method, mode), {}).get("mean_neg", 0)
                for layer in sorted_layers
            ]
            c = vec_colors[vn]
            short = vn.replace("reasoning_direction_", "rd:").replace("lr_weight_", "lr:")
            ax3.plot(sorted_layers, mp, marker="o", markersize=4,
                     color=c, linewidth=2, label=f"{short} (correct)")
            ax3.plot(sorted_layers, mn, marker="x", markersize=4,
                     color=c, linewidth=1.5, alpha=0.5)
        ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax3.set_ylabel("Mean Score", fontsize=10)
        ax3.set_xlabel("Layer", fontsize=10)
        ax3.set_xticks(sorted_layers)
        ax3.grid(True, alpha=0.3)

    method_label = "Cosine Similarity" if scoring_method == SCORING_COSINE else "LR Classifier (w·h+b)"
    fig.suptitle(
        f"Error Boundary Eval — {method_label}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=11,
                 style="italic", color="dimgray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")
    plt.close(fig)


def plot_lr_classifier_specific(all_metrics, sorted_layers, lr_vec_names,
                                 save_path, subtitle=None):
    """
    LR-specific figure showing accuracy, F1, precision, recall across layers
    for each LR vector in each extraction mode.
    """
    n_modes = len(EXTRACTION_MODES)
    palette = plt.cm.tab10.colors
    vec_colors = {vn: palette[i % len(palette)] for i, vn in enumerate(lr_vec_names)}

    clf_metrics_names = ["accuracy", "f1", "precision", "recall"]
    n_rows = len(clf_metrics_names)

    fig = plt.figure(figsize=(6 * n_modes, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_modes, hspace=0.4, wspace=0.3)

    for col_idx, mode in enumerate(EXTRACTION_MODES):
        for row_idx, metric_name in enumerate(clf_metrics_names):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            for vn in lr_vec_names:
                vals = [
                    all_metrics.get((vn, layer, SCORING_LR, mode), {}).get(metric_name, 0)
                    for layer in sorted_layers
                ]
                short = vn.replace("lr_weight_", "LR ")
                ax.plot(sorted_layers, vals, marker="o", markersize=4,
                        color=vec_colors[vn], linewidth=2, label=short)
            if metric_name == "accuracy":
                ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            ax.set_ylabel(metric_name.capitalize(), fontsize=10)
            if row_idx == 0:
                ax.set_title(f"Mode: {mode}", fontsize=12, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=10)
            ax.set_xticks(sorted_layers)
            ax.grid(True, alpha=0.3)
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("LR Classifier Metrics at Error Boundary",
                 fontsize=14, fontweight="bold", y=0.99)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", fontsize=11,
                 style="italic", color="dimgray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")
    plt.close(fig)


# ==========================================
# Print summary
# ==========================================
def print_summary(all_metrics, sorted_layers, vec_names, scoring_method):
    method_label = "COSINE" if scoring_method == SCORING_COSINE else "LR CLASSIFIER (w·h+b)"
    print(f"\n{'=' * 120}")
    print(f"  SCORING: {method_label}")
    print(f"  Positive class = CORRECT | Negative class = INCORRECT")
    print(f"{'=' * 120}")

    best_overall = {"score": -1, "choice": None}
    is_lr = (scoring_method == SCORING_LR)

    for mode in EXTRACTION_MODES:
        print(f"\n{'─' * 100}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'─' * 100}")

        for vn in vec_names:
            print(f"\n    {vn}")
            if is_lr:
                header = (f"    {'Layer':>5} | {'AUROC':>8} | {'Acc':>7} | "
                          f"{'F1':>7} | {'Prec':>7} | {'Recall':>7} | "
                          f"{'%Pred+':>7} | {'Cohen d':>9}")
            else:
                header = (f"    {'Layer':>5} | {'AUROC':>8} | {'Cohen d':>9} | "
                          f"{'Gap':>9} | {'p-value':>10} | {'Mean(+)':>9} | {'Mean(-)':>9}")
            print(header)
            print(f"    {'─' * (len(header) - 4)}")

            best_auroc_val, best_auroc_layer = -1, -1

            for layer in sorted_layers:
                m = all_metrics.get((vn, layer, scoring_method, mode), {})
                auroc = m.get("auroc", 0.5)

                if is_lr:
                    acc = m.get("accuracy", 0)
                    f1 = m.get("f1", 0)
                    prec = m.get("precision", 0)
                    rec = m.get("recall", 0)
                    fpp = m.get("frac_predicted_pos", 0)
                    d = m.get("cohens_d", 0)
                    print(f"    {layer:>5} | {auroc:>8.4f} | {acc:>7.4f} | "
                          f"{f1:>7.4f} | {prec:>7.4f} | {rec:>7.4f} | "
                          f"{fpp:>7.3f} | {d:>+9.4f}")
                else:
                    d = m.get("cohens_d", 0)
                    gap = m.get("gap", 0)
                    p = m.get("mann_whitney_p", 1.0)
                    mp = m.get("mean_pos", 0)
                    mn = m.get("mean_neg", 0)
                    print(f"    {layer:>5} | {auroc:>8.4f} | {d:>+9.4f} | "
                          f"{gap:>+9.4f} | {p:>10.2e} | {mp:>+9.4f} | {mn:>+9.4f}")

                if auroc > best_auroc_val:
                    best_auroc_val = auroc
                    best_auroc_layer = layer

            if best_auroc_val > best_overall["score"]:
                best_overall["score"] = best_auroc_val
                best_overall["choice"] = (vn, best_auroc_layer, mode, best_auroc_val)

            print(f"    → Best: layer {best_auroc_layer} (AUROC={best_auroc_val:.4f})")

    print(f"\n{'=' * 120}")
    if best_overall["choice"]:
        vn, l, mode, auc = best_overall["choice"]
        print(f"  ★ BEST [{method_label}]: {vn} @ layer {l}, mode={mode}, AUROC={auc:.4f}")
    print(f"{'=' * 120}\n")


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Error Boundary Evaluation (cosine + LR classifier)")
    print("  Correct vs Incorrect Reasoning — no GPU needed")
    print("=" * 70)
    print(f"  Raw activations:  {args.raw_dir}")
    print(f"  Vector file:      {args.vector_file}")
    print(f"  Dataset file:     {args.dataset_file}")
    print(f"  LR weights:       {args.lr_weights_file or '(none)'}")
    print(f"  LR metrics:       {args.lr_metrics_file or '(none)'}")
    print(f"  Target layers:    {args.target_layers}")
    print(f"  Window size:      ±{args.window_size} tokens")
    print()

    # ==========================================
    # Phase 1: Build boundary map
    # ==========================================
    print("Phase 1: Building boundary map...")
    boundary_map, shard_sizes, median_rel_pos = build_boundary_map(
        args.raw_dir, args.dataset_file
    )

    valid_samples = {
        sid: info for sid, info in boundary_map.items()
        if info["boundary_local_pos"] is not None
    }
    n_correct = sum(1 for v in valid_samples.values() if v["is_correct"])
    n_incorrect = sum(1 for v in valid_samples.values() if not v["is_correct"])
    print(f"  Valid: {len(valid_samples)} (correct={n_correct}, incorrect={n_incorrect})")

    if n_correct == 0 or n_incorrect == 0:
        print("ERROR: Need both correct and incorrect samples.")
        sys.exit(1)

    # ==========================================
    # Load all direction vectors + intercepts
    # ==========================================

    # Structure: direction_vectors[(layer, vec_name)] = Tensor[d_model]
    # Structure: intercepts[(layer, vec_name)] = float or None
    direction_vectors = {}
    intercepts = {}

    # --- Reasoning direction vectors (cosine only, no intercept) ---
    print("\nLoading reasoning direction vectors...")
    vector_data = torch.load(args.vector_file, map_location="cpu", weights_only=False)
    layers_dict = vector_data["layers"]

    for layer in args.target_layers:
        hook_key = f"blocks.{layer}.hook_out"
        if hook_key not in layers_dict:
            continue
        for vn in REASONING_VECTOR_NAMES:
            if vn in layers_dict[hook_key]:
                direction_vectors[(layer, vn)] = layers_dict[hook_key][vn].to(torch.float32)
                intercepts[(layer, vn)] = None  # no intercept for mean-diff vectors

    # --- LR weight vectors (have intercept for classifier scoring) ---
    lr_intercepts_loaded = {}
    if args.lr_weights_file and os.path.exists(args.lr_weights_file):
        print("Loading LR weight vectors...")
        lr_raw = torch.load(args.lr_weights_file, map_location="cpu", weights_only=False)

        # Load intercepts from metrics JSON
        if args.lr_metrics_file and os.path.exists(args.lr_metrics_file):
            print("Loading LR intercepts from metrics file...")
            with open(args.lr_metrics_file, "r") as f:
                lr_metrics_json = json.load(f)
            for layer_str, layer_data in lr_metrics_json.items():
                if layer_str == "cross_weight_similarities":
                    continue
                if isinstance(layer_data, dict):
                    for gran, gran_data in layer_data.items():
                        if isinstance(gran_data, dict) and "intercept_input_space" in gran_data:
                            lr_intercepts_loaded[(layer_str, gran)] = gran_data["intercept_input_space"]
            print(f"  Loaded {len(lr_intercepts_loaded)} intercept values")
        else:
            print("  No LR metrics file → LR classifier scoring disabled")

        for layer in args.target_layers:
            layer_key = str(layer)
            if layer_key not in lr_raw:
                continue
            for gran in LR_GRANULARITIES:
                if gran in lr_raw[layer_key]:
                    vn = f"lr_weight_{gran}"
                    direction_vectors[(layer, vn)] = lr_raw[layer_key][gran].to(torch.float32)
                    # Attach intercept if available
                    intercepts[(layer, vn)] = lr_intercepts_loaded.get(
                        (layer_key, gran), None
                    )

    available_layers = sorted(set(l for l, _ in direction_vectors.keys()))
    available_vecs = sorted(set(vn for _, vn in direction_vectors.keys()))
    n_with_intercept = sum(1 for v in intercepts.values() if v is not None)
    print(f"  Total: {len(direction_vectors)} (layer, vector) pairs")
    print(f"  With intercept (LR classifier enabled): {n_with_intercept}")
    print(f"  Vectors: {available_vecs}")

    # ==========================================
    # Phase 2-3: Extract & score, per layer
    # ==========================================
    # all_metrics[(vec_name, layer, scoring_method, mode)] = metrics_dict
    all_metrics = {}

    for layer in available_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n  Layer {layer}: extracting boundary activations...")

        sample_activations = extract_boundary_activations(
            args.raw_dir, hook_name, valid_samples, shard_sizes, args.window_size
        )
        print(f"    Extracted {len(sample_activations)} samples")

        layer_vecs = [vn for vn in available_vecs
                      if (layer, vn) in direction_vectors]

        for vn in layer_vecs:
            dvec = direction_vectors[(layer, vn)]
            intercept_val = intercepts.get((layer, vn), None)

            scores = compute_all_scores(
                sample_activations, dvec, valid_samples, intercept=intercept_val
            )

            for (scoring_method, mode), class_scores in scores.items():
                pos = np.array(class_scores["correct"])
                neg = np.array(class_scores["incorrect"])

                if len(pos) == 0 or len(neg) == 0:
                    continue

                # Always compute discriminability metrics (AUROC, Cohen's d, etc.)
                discrim = compute_discriminability_metrics(pos, neg)

                if scoring_method == SCORING_LR:
                    # Also compute classifier-specific metrics (acc, F1, etc.)
                    clf = compute_classifier_metrics(pos, neg, threshold=0.0)
                    # Merge: classifier metrics + Cohen's d from discriminability
                    merged = {**clf, "cohens_d": discrim["cohens_d"],
                              "gap": discrim["gap"],
                              "mean_pos": discrim["mean_pos"],
                              "mean_neg": discrim["mean_neg"]}
                    all_metrics[(vn, layer, scoring_method, mode)] = merged
                else:
                    all_metrics[(vn, layer, scoring_method, mode)] = discrim

        del sample_activations

    # ==========================================
    # Save results
    # ==========================================
    print("\nSaving results...")

    def safe_float(x):
        f = float(x)
        return None if (np.isnan(f) or np.isinf(f)) else f

    json_results = {
        "config": {
            "raw_dir": str(args.raw_dir),
            "vector_file": str(args.vector_file),
            "dataset_file": str(args.dataset_file),
            "window_size": args.window_size,
            "target_layers": args.target_layers,
            "median_relative_error_position": median_rel_pos,
            "n_correct_samples": n_correct,
            "n_incorrect_samples": n_incorrect,
        },
        "metrics": {},
    }

    for (vn, layer, sm, mode), m in all_metrics.items():
        key = f"{vn}__layer{layer}__{sm}__{mode}"
        json_results["metrics"][key] = {
            k: safe_float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }

    results_path = os.path.join(args.output_dir, "error_boundary_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Results saved → {results_path}")

    # ==========================================
    # Print summaries
    # ==========================================
    # Cosine summary (all vectors)
    cosine_vecs = [vn for vn in available_vecs
                   if any((vn, l, SCORING_COSINE, m) in all_metrics
                          for l in available_layers for m in EXTRACTION_MODES)]
    if cosine_vecs:
        print_summary(all_metrics, available_layers, cosine_vecs, SCORING_COSINE)

    # LR classifier summary (only LR vectors with intercepts)
    lr_vecs = [vn for vn in available_vecs
               if any((vn, l, SCORING_LR, m) in all_metrics
                      for l in available_layers for m in EXTRACTION_MODES)]
    if lr_vecs:
        print_summary(all_metrics, available_layers, lr_vecs, SCORING_LR)

    # ==========================================
    # Plots
    # ==========================================
    subtitle = (f"Window: ±{args.window_size} tokens | "
                f"Median error pos: {median_rel_pos:.2f} | "
                f"N correct={n_correct}, N incorrect={n_incorrect}")

    # Plot 1: Cosine similarity (all vectors)
    if cosine_vecs:
        plot_results(
            all_metrics, available_layers, cosine_vecs,
            os.path.join(args.output_dir, "error_boundary_cosine.png"),
            subtitle=subtitle, scoring_method=SCORING_COSINE,
        )

    # Plot 2: LR classifier AUROC/Cohen's d (LR vectors only)
    if lr_vecs:
        plot_results(
            all_metrics, available_layers, lr_vecs,
            os.path.join(args.output_dir, "error_boundary_lr_score.png"),
            subtitle=subtitle, scoring_method=SCORING_LR,
        )

        # Plot 3: LR classifier-specific metrics (accuracy, F1, precision, recall)
        plot_lr_classifier_specific(
            all_metrics, available_layers, lr_vecs,
            os.path.join(args.output_dir, "error_boundary_lr_classifier_metrics.png"),
            subtitle=subtitle,
        )

    # ==========================================
    # Side-by-side comparison table: cosine vs LR for LR vectors
    # ==========================================
    if lr_vecs:
        print(f"\n{'=' * 100}")
        print("  SIDE-BY-SIDE: Cosine vs LR Classifier AUROC (LR weight vectors)")
        print(f"{'=' * 100}")
        print(f"  {'Vector':>20} | {'Mode':>12} | {'Layer':>5} | "
              f"{'Cos AUROC':>10} | {'LR AUROC':>10} | {'LR Acc':>8} | {'LR F1':>8}")
        print(f"  {'─' * 90}")

        for vn in lr_vecs:
            for mode in EXTRACTION_MODES:
                for layer in available_layers:
                    cos_m = all_metrics.get((vn, layer, SCORING_COSINE, mode), {})
                    lr_m = all_metrics.get((vn, layer, SCORING_LR, mode), {})
                    cos_auc = cos_m.get("auroc", 0.5)
                    lr_auc = lr_m.get("auroc", 0.5)
                    lr_acc = lr_m.get("accuracy", 0)
                    lr_f1 = lr_m.get("f1", 0)

                    # Only print if at least one is interesting
                    if cos_auc > 0.55 or lr_auc > 0.55:
                        short = vn.replace("lr_weight_", "LR ")
                        print(f"  {short:>20} | {mode:>12} | {layer:>5} | "
                              f"{cos_auc:>10.4f} | {lr_auc:>10.4f} | "
                              f"{lr_acc:>8.4f} | {lr_f1:>8.4f}")

        print(f"{'=' * 100}\n")

    print("Done!")


if __name__ == "__main__":
    main()