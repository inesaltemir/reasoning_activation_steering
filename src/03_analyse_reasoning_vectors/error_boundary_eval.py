"""
Error Boundary Evaluation — Correct vs Incorrect Reasoning
============================================================

Tests whether reasoning direction vectors carry a genuine reasoning-quality
signal by checking discriminability at the *error boundary* in ProcessBench
samples.  Unlike layer_selection_eval.py (which separates reasoning text from
non-reasoning text — essentially topic detection), this script tests the
actual research question:

    Can the vector distinguish correct reasoning from incorrect reasoning?

NO MODEL LOADING, NO GPU, NO FORWARD PASSES.
All activations are read from disk (raw_activations/ produced by
run_fw_pass_with_step_averaging_storage.py).

Algorithm
---------
1.  Load metadata from raw_activations/ to identify, for every ProcessBench
    sample, the per-token correctness labels (is_correct, step_idx).

2.  For INCORRECT samples (label ≥ 0): locate the *error boundary* — the
    first token where is_correct transitions from True → False.  Record its
    relative position  r_i = boundary_pos / total_reasoning_tokens.
    ==> error is not nec at the beginning of the first incorrect step, 
    should perhaps implement moving window (look at beginning middle and end?)
    when does the model become aware there is an error in the step?

3.  For CORRECT samples (label == -1): use a *positional control* at the
    median relative position of the incorrect-sample boundaries.

4.  Extract activations at the boundary / control position in three modes:
      • exact          — single token at the boundary
      • window_mean    — mean of ±W tokens around the boundary
      • window_max_cos — per-token cosine sim in ±W window, take max

5.  Compute cosine similarity with every reasoning direction vector
    (token/step/sample, raw/cleaned) and every LR weight vector (if provided).

6.  Compute discriminability metrics (Cohen's d, AUROC, Mann-Whitney p)
    treating correct-sample scores as positive and incorrect-sample scores
    as negative.

7.  Plot and save.

Usage
-----
python3 error_boundary_eval.py 
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
from sklearn.metrics import roc_auc_score
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
    p.add_argument("--lr_weights_file", type=str, default=None,
                   help="Optional: path to lr_learned_weights.pt for comparison")
    p.add_argument("--target_layers", type=int, nargs="+",
                   default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--window_size", type=int, default=5,
                   help="Half-width of the token window around the error boundary "
                        "(default 5 → 11-token window)")
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/error_boundary_eval")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ==========================================
# Reasoning vector names to look for
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

# Extraction modes
EXTRACTION_MODES = ["exact", "window_mean", "window_max_cos"]


# ==========================================
# Discriminability metrics (same as layer_selection_eval.py)
# ==========================================
def compute_discriminability_metrics(positive_scores, negative_scores):
    """
    Compute discrimination metrics.
    Positive = correct samples (expected higher cosine sim).
    Negative = incorrect samples (expected lower cosine sim at error boundary).
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

    mean_pos = np.mean(positive_scores)
    mean_neg = np.mean(negative_scores)
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


# ==========================================
# Phase 1: Build per-sample boundary map
# ==========================================
def build_boundary_map(raw_dir, dataset_file):
    """
    Load all token metadata from raw_activations/ and the ProcessBench
    dataset.  For each sample, determine:
      - Whether it is correct or incorrect
      - The error boundary token position (for incorrect samples)
      - The total number of reasoning tokens

    Returns
    -------
    boundary_map : dict
        sample_idx → {
            "is_correct": bool,
            "boundary_local_pos": int or None,    # within this sample's token range
            "total_tokens": int,
            "relative_pos": float or None,         # boundary_local_pos / total_tokens
            "global_start_row": int,               # start row in the concatenated shard space
            "label": int,                          # ProcessBench label (-1 = all correct)
        }
    shard_sizes : list[int]
        Number of token-rows in each shard (needed to map global rows → shards).
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    sample_index = index["sample_index"]  # [(start_row, num_tokens, sample_idx), ...]
    num_shards = index["num_shards"]

    # --- Load ProcessBench labels ---
    dataset_labels = {}  # sample_idx → label (int)
    dataset_correct = {}  # sample_idx → final_answer_correct (bool)
    with open(dataset_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())
            dataset_labels[i] = sample.get("label", -1)
            dataset_correct[i] = sample.get("final_answer_correct", None)

    # --- Load ALL metadata (concatenated across shards) ---
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

    # --- Build boundary map ---
    print("  Building boundary map...")
    boundary_map = {}

    for start_row, num_tokens, sample_idx in sample_index:
        sample_meta = all_meta[start_row: start_row + num_tokens]

        pb_label = dataset_labels.get(sample_idx, -1)
        pb_correct = dataset_correct.get(sample_idx, None)

        # Determine if sample is fully correct
        # A sample is "correct" if label == -1 AND final_answer_correct is True
        is_correct = (pb_label == -1) and (pb_correct is True)

        # Find error boundary: first token where is_correct transitions to False
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

    # --- Summary ---
    n_correct = sum(1 for v in boundary_map.values() if v["is_correct"])
    n_incorrect = sum(1 for v in boundary_map.values() if not v["is_correct"])
    n_with_boundary = sum(
        1 for v in boundary_map.values()
        if v["boundary_local_pos"] is not None
    )

    print(f"  Boundary map built: {len(boundary_map)} samples")
    print(f"    Correct (positive class): {n_correct}")
    print(f"    Incorrect (negative class): {n_incorrect}")
    print(f"    Incorrect with detected boundary: {n_with_boundary}")

    # --- Compute median relative position for control ---
    relative_positions = [
        v["relative_pos"]
        for v in boundary_map.values()
        if v["relative_pos"] is not None
    ]
    if relative_positions:
        median_rel_pos = float(np.median(relative_positions))
        mean_rel_pos = float(np.mean(relative_positions))
        std_rel_pos = float(np.std(relative_positions))
        print(f"    Error boundary relative position: "
              f"median={median_rel_pos:.3f}, mean={mean_rel_pos:.3f}, "
              f"std={std_rel_pos:.3f}")
    else:
        median_rel_pos = 0.5
        print(f"    WARNING: no error boundaries found, using default median=0.5")

    # --- Assign control positions to correct samples ---
    for sample_idx, info in boundary_map.items():
        if info["is_correct"]:
            control_pos = int(median_rel_pos * info["total_tokens"])
            # Clamp to valid range
            control_pos = max(0, min(control_pos, info["total_tokens"] - 1))
            info["boundary_local_pos"] = control_pos
            info["relative_pos"] = control_pos / max(info["total_tokens"], 1)

    return boundary_map, shard_sizes, median_rel_pos


# ==========================================
# Phase 2: Extract boundary activations
# ==========================================
def extract_boundary_activations(
    raw_dir, hook_name, boundary_map, shard_sizes, window_size
):
    """
    For one layer (hook_name), load activations from shards and extract
    the boundary window for every sample.

    Returns
    -------
    sample_activations : dict
        sample_idx → {
            "exact": Tensor[d_model],             # single boundary token
            "window": Tensor[window_len, d_model], # ±W tokens
            "window_positions": list[int],          # global row indices of window tokens
        }
    """
    raw_dir = Path(raw_dir)
    safe_name = hook_name.replace(".", "_")
    num_shards = len(shard_sizes)

    # Precompute shard boundaries (cumulative sum)
    shard_boundaries = np.zeros(num_shards + 1, dtype=np.int64)
    for i, sz in enumerate(shard_sizes):
        shard_boundaries[i + 1] = shard_boundaries[i] + sz

    # --- Identify all needed global rows per sample ---
    # For each sample, we need rows in [boundary_pos - W, boundary_pos + W]
    # clamped to [0, total_tokens - 1], offset by global_start_row.
    needed_rows = {}  # global_row → list of (sample_idx, local_window_idx)
    sample_window_info = {}  # sample_idx → list of global_rows in order

    for sample_idx, info in boundary_map.items():
        bpos = info["boundary_local_pos"]
        if bpos is None:
            continue  # skip samples without a boundary (shouldn't happen after Phase 1)

        total = info["total_tokens"]
        start = info["global_start_row"]

        # Window range within this sample's tokens
        win_start = max(0, bpos - window_size)
        win_end = min(total - 1, bpos + window_size)

        window_global_rows = []
        for local_pos in range(win_start, win_end + 1):
            grow = start + local_pos
            window_global_rows.append(grow)
            if grow not in needed_rows:
                needed_rows[grow] = []
            needed_rows[grow].append((sample_idx, len(window_global_rows) - 1))

        sample_window_info[sample_idx] = {
            "global_rows": window_global_rows,
            "boundary_offset_in_window": bpos - win_start,  # which index in window is the exact boundary
        }

    # --- Load shards and extract needed rows ---
    # Group needed rows by shard for efficient loading
    rows_by_shard = defaultdict(list)
    for grow in needed_rows:
        shard_id = int(np.searchsorted(shard_boundaries[1:], grow, side="right"))
        local_row = grow - int(shard_boundaries[shard_id])
        rows_by_shard[shard_id].append((grow, local_row))

    # Extract activations
    row_to_activation = {}  # global_row → Tensor[d_model]

    for shard_id in sorted(rows_by_shard.keys()):
        shard_path = raw_dir / safe_name / f"shard_{shard_id:04d}.pt"
        acts = torch.load(shard_path, weights_only=False).to(torch.float32)

        for grow, local_row in rows_by_shard[shard_id]:
            row_to_activation[grow] = acts[local_row]

        del acts

    # --- Assemble per-sample results ---
    sample_activations = {}

    for sample_idx, win_info in sample_window_info.items():
        global_rows = win_info["global_rows"]
        boundary_idx = win_info["boundary_offset_in_window"]

        window_acts = torch.stack([
            row_to_activation[gr] for gr in global_rows
        ])  # (window_len, d_model)

        exact_act = window_acts[boundary_idx]  # (d_model,)

        sample_activations[sample_idx] = {
            "exact": exact_act,
            "window": window_acts,
            "boundary_idx_in_window": boundary_idx,
        }

    return sample_activations


# ==========================================
# Phase 3: Compute cosine similarities
# ==========================================
def compute_cosine_scores(sample_activations, direction_vec, boundary_map):
    """
    Compute cosine similarity between each sample's boundary activation
    and a reasoning direction vector, in three extraction modes.

    Returns
    -------
    scores : dict
        {
            "exact":          {"correct": [...], "incorrect": [...]},
            "window_mean":    {"correct": [...], "incorrect": [...]},
            "window_max_cos": {"correct": [...], "incorrect": [...]},
        }
    """
    scores = {
        mode: {"correct": [], "incorrect": []}
        for mode in EXTRACTION_MODES
    }

    dvec = direction_vec.to(torch.float32)
    dvec_2d = dvec.unsqueeze(0)  # (1, d_model) for batch cosine sim

    for sample_idx, acts in sample_activations.items():
        info = boundary_map[sample_idx]
        label_key = "correct" if info["is_correct"] else "incorrect"

        # --- exact: single boundary token ---
        cos_exact = F.cosine_similarity(
            acts["exact"].unsqueeze(0), dvec_2d, dim=-1
        ).item()
        scores["exact"][label_key].append(cos_exact)

        # --- window_mean: mean of window → cosine sim ---
        window_mean = acts["window"].mean(dim=0)  # (d_model,)
        cos_wmean = F.cosine_similarity(
            window_mean.unsqueeze(0), dvec_2d, dim=-1
        ).item()
        scores["window_mean"][label_key].append(cos_wmean)

        # --- window_max_cos: per-token cosine, take max ---
        per_token_cos = F.cosine_similarity(
            acts["window"], dvec_2d.expand(acts["window"].shape[0], -1), dim=-1
        )  # (window_len,)
        cos_wmax = per_token_cos.max().item()
        scores["window_max_cos"][label_key].append(cos_wmax)

    return scores


# ==========================================
# Plotting
# ==========================================
def plot_error_boundary_results(
    all_metrics, sorted_layers, vec_names, save_path, subtitle=None
):
    """
    Multi-panel figure.
    Columns = extraction modes (exact, window_mean, window_max_cos)
    Row 1 = AUROC across layers
    Row 2 = Cohen's d across layers
    Row 3 = Mean cosine sim (correct vs incorrect) across layers
    """
    n_modes = len(EXTRACTION_MODES)

    # Assign colours: cycle through a palette
    palette = plt.cm.tab10.colors
    vec_colors = {vn: palette[i % len(palette)] for i, vn in enumerate(vec_names)}

    fig = plt.figure(figsize=(6 * n_modes, 14))
    gs = gridspec.GridSpec(3, n_modes, hspace=0.35, wspace=0.3)

    for col_idx, mode in enumerate(EXTRACTION_MODES):
        # --- Row 1: AUROC ---
        ax1 = fig.add_subplot(gs[0, col_idx])
        for vn in vec_names:
            aurocs = [
                all_metrics.get((vn, layer, mode), {}).get("auroc", 0.5)
                for layer in sorted_layers
            ]
            ax1.plot(sorted_layers, aurocs, marker="o", markersize=4,
                     color=vec_colors[vn], linewidth=2,
                     label=vn.replace("reasoning_direction_", "").replace("lr_weight_", "LR "))
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.set_ylabel("AUROC", fontsize=10)
        ax1.set_title(f"Mode: {mode}", fontsize=12, fontweight="bold")
        ax1.set_ylim(0.3, 1.0)
        ax1.set_xticks(sorted_layers)
        ax1.grid(True, alpha=0.3)
        if col_idx == 0:
            ax1.legend(fontsize=6, loc="best")

        # --- Row 2: Cohen's d ---
        ax2 = fig.add_subplot(gs[1, col_idx])
        for vn in vec_names:
            ds = [
                all_metrics.get((vn, layer, mode), {}).get("cohens_d", 0)
                for layer in sorted_layers
            ]
            ax2.plot(sorted_layers, ds, marker="s", markersize=4,
                     color=vec_colors[vn], linewidth=2)
        ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Large (0.8)")
        ax2.set_ylabel("Cohen's d", fontsize=10)
        ax2.set_xlabel("Layer", fontsize=10)
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, alpha=0.3)

        # --- Row 3: Mean cosine sim ---
        ax3 = fig.add_subplot(gs[2, col_idx])
        for vn in vec_names:
            mp = [
                all_metrics.get((vn, layer, mode), {}).get("mean_pos", 0)
                for layer in sorted_layers
            ]
            mn = [
                all_metrics.get((vn, layer, mode), {}).get("mean_neg", 0)
                for layer in sorted_layers
            ]
            c = vec_colors[vn]
            short = vn.replace("reasoning_direction_", "").replace("lr_weight_", "LR ")
            ax3.plot(sorted_layers, mp, marker="o", markersize=4,
                     color=c, linewidth=2, label=f"{short} (correct)")
            ax3.plot(sorted_layers, mn, marker="x", markersize=4,
                     color=c, linewidth=1.5, alpha=0.5)
        ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax3.set_ylabel("Mean Cosine Sim", fontsize=10)
        ax3.set_xlabel("Layer", fontsize=10)
        ax3.set_xticks(sorted_layers)
        ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Error Boundary Eval: Correct vs Incorrect Reasoning",
        fontsize=14, fontweight="bold", y=0.98,
    )
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=11,
                 style="italic", color="dimgray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")


# ==========================================
# Print summary
# ==========================================
def print_summary(all_metrics, sorted_layers, vec_names):
    """Print a table of AUROC / Cohen's d for each (vector, mode) combination."""
    print("\n" + "=" * 120)
    print("  ERROR BOUNDARY EVALUATION SUMMARY")
    print("  Positive class = CORRECT samples | Negative class = INCORRECT samples")
    print("=" * 120)

    best_overall = {"score": -1, "choice": None}

    for mode in EXTRACTION_MODES:
        print(f"\n{'─' * 100}")
        print(f"  Extraction mode: {mode.upper()}")
        print(f"{'─' * 100}")

        for vn in vec_names:
            print(f"\n    Vector: {vn}")
            header = f"    {'Layer':>5} | {'AUROC':>8} | {'Cohen d':>9} | {'Gap':>9} | {'p-value':>10} | {'Mean(+)':>9} | {'Mean(-)':>9}"
            print(header)
            print(f"    {'─' * (len(header) - 4)}")

            best_auroc_layer = -1
            best_auroc_val = -1

            for layer in sorted_layers:
                m = all_metrics.get((vn, layer, mode), {})
                auroc = m.get("auroc", 0.5)
                d = m.get("cohens_d", 0)
                gap = m.get("gap", 0)
                p = m.get("mann_whitney_p", 1.0)
                mp = m.get("mean_pos", 0)
                mn = m.get("mean_neg", 0)

                marker = ""
                if auroc > best_auroc_val:
                    best_auroc_val = auroc
                    best_auroc_layer = layer

                print(f"    {layer:>5} | {auroc:>8.4f} | {d:>+9.4f} | "
                      f"{gap:>+9.4f} | {p:>10.2e} | {mp:>+9.4f} | {mn:>+9.4f}")

            # Mark best layer
            if best_auroc_val > best_overall["score"]:
                best_overall["score"] = best_auroc_val
                best_overall["choice"] = (vn, best_auroc_layer, mode, best_auroc_val)

            print(f"    → Best layer for {vn}/{mode}: layer {best_auroc_layer} "
                  f"(AUROC={best_auroc_val:.4f})")

    print(f"\n{'=' * 120}")
    if best_overall["choice"]:
        vn, l, mode, auc = best_overall["choice"]
        m = all_metrics.get((vn, l, mode), {})
        d = m.get("cohens_d", 0)
        print(f"  ★ BEST OVERALL: {vn} at layer {l}, mode={mode}")
        print(f"    AUROC = {auc:.4f}, Cohen's d = {d:+.4f}")
    print(f"{'=' * 120}\n")


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Error Boundary Evaluation")
    print("  Correct vs Incorrect Reasoning (no GPU needed)")
    print("=" * 70)
    print(f"  Raw activations:  {args.raw_dir}")
    print(f"  Vector file:      {args.vector_file}")
    print(f"  Dataset file:     {args.dataset_file}")
    print(f"  LR weights:       {args.lr_weights_file or '(none)'}")
    print(f"  Target layers:    {args.target_layers}")
    print(f"  Window size:      ±{args.window_size} tokens")
    print(f"  Output dir:       {args.output_dir}")
    print()

    # ==========================================
    # Phase 1: Build boundary map
    # ==========================================
    print("Phase 1: Building boundary map from metadata...")
    boundary_map, shard_sizes, median_rel_pos = build_boundary_map(
        args.raw_dir, args.dataset_file
    )

    # Filter to samples that have a boundary position assigned
    valid_samples = {
        sid: info for sid, info in boundary_map.items()
        if info["boundary_local_pos"] is not None
    }
    n_correct = sum(1 for v in valid_samples.values() if v["is_correct"])
    n_incorrect = sum(1 for v in valid_samples.values() if not v["is_correct"])
    print(f"  Valid samples for evaluation: {len(valid_samples)} "
          f"(correct={n_correct}, incorrect={n_incorrect})")

    if n_correct == 0 or n_incorrect == 0:
        print("ERROR: Need both correct and incorrect samples. Exiting.")
        sys.exit(1)

    # ==========================================
    # Load reasoning direction vectors
    # ==========================================
    print("\nLoading reasoning direction vectors...")
    vector_data = torch.load(args.vector_file, map_location="cpu", weights_only=False)
    layers_dict = vector_data["layers"]

    direction_vectors = {}  # (layer, vec_name) → Tensor[d_model]
    for layer in args.target_layers:
        hook_key = f"blocks.{layer}.hook_out"
        if hook_key not in layers_dict:
            print(f"  Warning: {hook_key} not in vector file, skipping layer {layer}")
            continue
        for vn in REASONING_VECTOR_NAMES:
            if vn in layers_dict[hook_key]:
                direction_vectors[(layer, vn)] = layers_dict[hook_key][vn].to(
                    torch.float32
                )

    # Optionally load LR weight vectors
    if args.lr_weights_file and os.path.exists(args.lr_weights_file):
        print("Loading LR classifier weight vectors...")
        lr_raw = torch.load(args.lr_weights_file, map_location="cpu", weights_only=False)
        for layer in args.target_layers:
            layer_key = str(layer)
            if layer_key not in lr_raw:
                continue
            for gran in LR_GRANULARITIES:
                if gran in lr_raw[layer_key]:
                    vn = f"lr_weight_{gran}"
                    direction_vectors[(layer, vn)] = lr_raw[layer_key][gran].to(
                        torch.float32
                    )

    available_layers = sorted(set(l for l, _ in direction_vectors.keys()))
    available_vecs = sorted(set(vn for _, vn in direction_vectors.keys()))
    print(f"  Loaded {len(direction_vectors)} (layer, vector) pairs")
    print(f"  Layers: {available_layers}")
    print(f"  Vector types: {available_vecs}")

    # ==========================================
    # Phase 2-3: Extract activations & compute scores, per layer
    # ==========================================
    # all_metrics[(vec_name, layer, mode)] = metrics_dict
    all_metrics = {}

    for layer in available_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n  Layer {layer}: extracting boundary activations...")

        sample_activations = extract_boundary_activations(
            args.raw_dir, hook_name, valid_samples, shard_sizes, args.window_size
        )
        print(f"    Extracted {len(sample_activations)} samples")

        # For each vector type available at this layer
        layer_vecs = [
            vn for vn in available_vecs if (layer, vn) in direction_vectors
        ]

        for vn in layer_vecs:
            dvec = direction_vectors[(layer, vn)]
            scores = compute_cosine_scores(sample_activations, dvec, valid_samples)

            for mode in EXTRACTION_MODES:
                pos = np.array(scores[mode]["correct"])
                neg = np.array(scores[mode]["incorrect"])

                if len(pos) > 0 and len(neg) > 0:
                    metrics = compute_discriminability_metrics(pos, neg)
                    all_metrics[(vn, layer, mode)] = metrics

        del sample_activations  # free memory before next layer

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

    for (vn, layer, mode), m in all_metrics.items():
        key = f"{vn}__layer{layer}__{mode}"
        json_results["metrics"][key] = {
            k: safe_float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }

    results_path = os.path.join(args.output_dir, "error_boundary_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Results saved → {results_path}")

    # ==========================================
    # Print summary
    # ==========================================
    print_summary(all_metrics, available_layers, available_vecs)

    # ==========================================
    # Plot
    # ==========================================
    subtitle = (f"Window: ±{args.window_size} tokens | "
                f"Median error position: {median_rel_pos:.2f} | "
                f"N correct={n_correct}, N incorrect={n_incorrect}")
    plot_path = os.path.join(args.output_dir, "error_boundary_eval.png")
    plot_error_boundary_results(
        all_metrics, available_layers, available_vecs, plot_path, subtitle
    )

    # ==========================================
    # Bonus: per-mode comparison bar chart
    # ==========================================
    # For each vector, compare AUROC across modes at the best layer
    print("\n  Mode comparison (AUROC at best layer per vector):")
    for vn in available_vecs:
        for mode in EXTRACTION_MODES:
            best_auc = max(
                (all_metrics.get((vn, l, mode), {}).get("auroc", 0.5)
                 for l in available_layers),
                default=0.5,
            )
            best_layer = max(
                available_layers,
                key=lambda l: all_metrics.get((vn, l, mode), {}).get("auroc", 0.5),
            )
            print(f"    {vn:>45} | {mode:>15} | "
                  f"layer {best_layer:>2} | AUROC={best_auc:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
