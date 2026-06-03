"""
Evaluate LR Classifier on Math-Shepherd Stored Activations
===========================================================

This script evaluates the logistic regression weight vectors from
`lr_learned_weights.pt` (produced by train_lr_classifier_streaming_extended.py)
on the Math-Shepherd step-averaged activations produced by
`run_fw_pass_with_step_averaging_storage.py`.

ONLY CLASSIFIER MODE is used:
  score = w_raw · activation + intercept_input_space
  Binary classifier metrics: accuracy, F1, precision, recall, AUROC on the logit.

Two evaluation granularities:
  1. STEP-LEVEL  — each step's mean activation is one data point.
                   Label: is the step correct (True) or incorrect (False)?
                   Groups: stratified by sample_idx for reporting.

  2. SAMPLE-LEVEL — each sample's mean activation (across all reasoning tokens)
                    is one data point.
                    Label: is the sample fully correct (label == -1) or not?

Memory strategy (mirrors train_lr_classifier_streaming_extended.py):
  - Shard files are loaded ONE AT A TIME via load_one_shard().
  - Step-level aggregation keeps only running (sum, count) dicts in RAM,
    never the full token matrix.  Peak RAM ≈ one shard + the accumulated
    step means (both well under 4 GB).
  - Sample-level activations are read from per_sample_means already stored
    in the .pt summary file (no shard scanning needed).

Usage:
  python3 eval_lr_classifier_math_shepherd.py \\
    --activations_file /path/to/reasoning_vectors_..._with_steps_avg_storage.pt \\
    --lr_weights_file  /path/to/lr_learned_weights.pt \\
    --lr_metrics_file  /path/to/lr_classifier_results.json \\
    --target_layers 18 19 20 21 22 23 24 25 26 27 28 \\
    --output_dir   /path/to/output/dir \\
    --gpu 0
"""

import os
import sys
import argparse
import json
import warnings
import gc
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. Parse arguments FIRST (before CUDA imports)
# ==========================================
parser = argparse.ArgumentParser(
    description="Evaluate LR classifier on Math-Shepherd stored activations (classifier mode only)."
)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument(
    "--activations_file", type=str,
    default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/math-shepherd/reasoning_vectors_Qwen3-8B_math-shepherd_with_steps_avg_storage.pt",
    help="Path to the .pt file produced by run_fw_pass_with_step_averaging_storage.py",
)
parser.add_argument(
    "--lr_weights_file", type=str,
    default="/home/ines/Reasoning-activations/results/lr_classifier_no_leak/lr_learned_weights.pt",
    help="Path to lr_learned_weights.pt",
)
parser.add_argument(
    "--lr_metrics_file", type=str,
    default="/home/ines/Reasoning-activations/results/lr_classifier_no_leak/lr_classifier_results.json",
    help="Path to lr_classifier_results.json (for intercepts).",
)
parser.add_argument(
    "--target_layers", type=int, nargs="+", default=list(range(18, 29)),
    help="Layers to evaluate.",
)
parser.add_argument(
    "--output_dir", type=str,
    default="/home/ines/Reasoning-activations/results/lr_classifier_no_leak/eval_math_shepherd",
    help="Directory for output JSON and plots.",
)
parser.add_argument(
    "--raw_activations_dir", type=str, default=None,
    help="Override the raw_activations_dir stored in the .pt metadata. "
         "Useful if files were moved.",
)
parser.add_argument(
    "--granularities", type=str, nargs="+", default=None,
    help="Which LR granularities to evaluate (e.g. step sample token). "
         "Default: all found in lr_weights_file.",
)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==========================================
# 2. Imports (after CUDA env is set)
# ==========================================
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# Shard loader — identical to train_lr_classifier_streaming_extended.py
# Loads ONE shard at a time; caller is responsible for del + gc.collect().
# ==========================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    """Load a single shard + its metadata. Returns float32 tensor + list[dict]."""
    safe = hook_name.replace(".", "_")
    acts = torch.load(
        raw_dir / safe / f"shard_{shard_id:04d}.pt",
        weights_only=False,
    ).to(torch.float32)
    meta = torch.load(
        raw_dir / f"meta_shard_{shard_id:04d}.pt",
        weights_only=False,
    )
    return acts, meta


# ==========================================
# STEP-LEVEL streaming aggregation
# Mirrors run_step_level() in train_lr_classifier_streaming_extended.py.
# Never holds more than one shard in RAM; accumulates only running sums.
# ==========================================
def aggregate_step_activations(raw_dir: Path, hook_name: str,
                                num_shards: int, d_model: int):
    """Stream all shards, accumulate per-(sample_idx, step_idx) sums.

    Returns
    -------
    X      : (S, d_model) float32 — one row per step
    y      : (S,)         int32   — 1=correct, 0=incorrect
    groups : (S,)         int32   — sample_idx (for group-stratified reporting)
    """
    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)   # any False token flips the step

    print(f"    Streaming {num_shards} shards for step aggregation ({hook_name})...")
    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)

        for i, m in enumerate(meta):
            if m["step_idx"] < 0 or m["is_correct"] is None:
                continue
            key = (m["sample_idx"], m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            _ = step_labels[key]                  # ensure key exists (default True)
            if m["is_correct"] is False:
                step_labels[key] = False

        del acts, acts_np, meta
        gc.collect()

    # Build X, y, groups from accumulated dicts
    keys   = sorted(step_sums.keys())
    X      = np.stack([step_sums[k] / step_counts[k] for k in keys]).astype(np.float32)
    y      = np.array([1 if step_labels[k] else 0 for k in keys], dtype=np.int32)
    groups = np.array([k[0] for k in keys], dtype=np.int32)

    n_correct   = int(y.sum())
    n_incorrect = int(len(y) - y.sum())
    print(f"    Steps: {len(keys):,}  |  Correct: {n_correct:,}  |  Incorrect: {n_incorrect:,}")

    del step_sums, step_counts, step_labels
    gc.collect()

    return X, y, groups


# ==========================================
# Classifier score computation
# ==========================================
def compute_scores(X: np.ndarray, weight_vec: torch.Tensor,
                   intercept: float) -> np.ndarray:
    """score = w · h + b  for each row in X."""
    w = weight_vec.numpy().astype(np.float32)
    return X @ w + intercept          # (N,)


# ==========================================
# Metrics  — identical to eval_lr_vectors_as_classifiers_v2.py
# ==========================================
def compute_classifier_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                                threshold: float = 0.0) -> dict:
    """Binary classification metrics (score > threshold → positive)."""
    y_pred = (y_scores > threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5

    if len(np.unique(y_true)) < 2:
        return {
            "accuracy": float(np.mean(y_pred == y_true)),
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "auroc": float(auroc), "threshold": threshold,
            "n_pos": int(y_true.sum()), "n_neg": int((~y_true.astype(bool)).sum()),
            "frac_predicted_pos": float(y_pred.mean()),
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc":              float(auroc),
        "threshold":          threshold,
        "n_pos":              int(y_true.sum()),
        "n_neg":              int((~y_true.astype(bool)).sum()),
        "frac_predicted_pos": float(y_pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ==========================================
# Plotting — mirrors eval_lr_vectors_as_classifiers_v2.py
# ==========================================
def plot_classifier_results(clf_metrics: dict, sorted_layers: list,
                             granularities: list, eval_granularities: list,
                             save_path: str, subtitle: str = ""):
    """Accuracy / Precision / Recall / F1 / AUROC across layers."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auroc"]
    metric_labels   = ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"]

    colors  = {"token": "#2196F3", "step": "#4CAF50", "sample": "#FF9800"}
    markers = {"token": "o",       "step": "s",        "sample": "D"}

    n_cols = len(eval_granularities)
    n_rows = len(metrics_to_plot)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    for col_idx, eval_gran in enumerate(eval_granularities):
        for row_idx, (metric, mlabel) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[row_idx, col_idx]

            for lr_gran in granularities:
                vals = [
                    clf_metrics.get((lr_gran, layer, eval_gran), {}).get(metric, 0.0)
                    for layer in sorted_layers
                ]
                ax.plot(sorted_layers, vals,
                        marker=markers.get(lr_gran, "o"),
                        markersize=5,
                        color=colors.get(lr_gran, "gray"),
                        linewidth=2,
                        label=f"LR {lr_gran}")

            ax.set_ylabel(mlabel, fontsize=10)
            if row_idx == 0:
                ax.set_title(f"Eval: {eval_gran}-level", fontsize=12, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=10)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(sorted_layers)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "LR Classifier on Math-Shepherd: Classifier Mode (Dot Product + Intercept)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    if subtitle:
        fig.text(0.5, 0.975, subtitle, ha="center", fontsize=10,
                 style="italic", color="dimgray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Classifier plot saved → {save_path}")


def plot_confusion_matrices(clf_metrics: dict, sorted_layers: list,
                             granularities: list, eval_granularities: list,
                             save_path: str):
    """Stacked bar showing TP/TN/FP/FN composition across layers."""
    n_rows = len(granularities)
    n_cols = len(eval_granularities)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for r_idx, lr_gran in enumerate(granularities):
        for c_idx, eval_gran in enumerate(eval_granularities):
            ax = axes[r_idx, c_idx]

            tp_pct, tn_pct, fp_pct, fn_pct = [], [], [], []
            for layer in sorted_layers:
                m     = clf_metrics.get((lr_gran, layer, eval_gran), {})
                total = m.get("n_pos", 0) + m.get("n_neg", 0)
                if total == 0:
                    tp_pct.append(0); tn_pct.append(0)
                    fp_pct.append(0); fn_pct.append(0)
                    continue
                tp_pct.append(100 * m.get("tp", 0) / total)
                tn_pct.append(100 * m.get("tn", 0) / total)
                fp_pct.append(100 * m.get("fp", 0) / total)
                fn_pct.append(100 * m.get("fn", 0) / total)

            ax.bar(sorted_layers, tp_pct, label="True Positive (TP)",  color="#4CAF50")
            ax.bar(sorted_layers, tn_pct, bottom=tp_pct, label="True Negative (TN)", color="#2196F3")
            fp_bottom = [a + b for a, b in zip(tp_pct, tn_pct)]
            ax.bar(sorted_layers, fp_pct, bottom=fp_bottom, label="False Positive (FP)", color="#F44336")
            fn_bottom = [a + b + c for a, b, c in zip(tp_pct, tn_pct, fp_pct)]
            ax.bar(sorted_layers, fn_pct, bottom=fn_bottom, label="False Negative (FN)", color="#FF9800")

            ax.set_title(f"LR {lr_gran.upper()} | Eval: {eval_gran}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Data Distribution (%)", fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_xticks(sorted_layers)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=9)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7, loc="lower left",
                          bbox_to_anchor=(0, 1.02), ncol=4)

    fig.suptitle("Confusion Matrix Composition Across Layers (Math-Shepherd)",
                 fontsize=13, fontweight="bold", y=0.998)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved → {save_path}")


# ==========================================
# Summary printer — mirrors print_recommendation()
# ==========================================
def print_summary(clf_metrics: dict, sorted_layers: list,
                  granularities: list, eval_granularities: list):
    print("\n" + "=" * 120)
    print("  LR CLASSIFIER — MATH-SHEPHERD EVALUATION SUMMARY")
    print("=" * 120)

    best_score  = -1
    best_choice = None

    for lr_gran in granularities:
        print(f"\n{'─' * 100}")
        print(f"  LR Granularity: {lr_gran.upper()}")
        print(f"{'─' * 100}")

        col_w = 16
        header = f"  {'Layer':>5}"
        for eg in eval_granularities:
            header += f" | {(eg+' AUROC'):>{col_w}} | {(eg+' Acc'):>{col_w}} | {(eg+' F1'):>{col_w}}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for layer in sorted_layers:
            row = f"  {layer:>5}"
            auroc_vals = []
            for eg in eval_granularities:
                m     = clf_metrics.get((lr_gran, layer, eg), {})
                auroc = m.get("auroc", 0.5)
                acc   = m.get("accuracy", 0.0)
                f1    = m.get("f1", 0.0)
                row  += f" | {auroc:>{col_w}.4f} | {acc:>{col_w}.4f} | {f1:>{col_w}.4f}"
                auroc_vals.append(auroc)
            print(row)

            avg_auroc = float(np.mean(auroc_vals)) if auroc_vals else 0.5
            if avg_auroc > best_score:
                best_score  = avg_auroc
                best_choice = (lr_gran, layer, avg_auroc)

    print(f"\n{'=' * 120}")
    if best_choice:
        g, l, auc = best_choice
        print(f"  ★ BEST (avg AUROC across eval granularities): LR '{g}' at layer {l}  |  Avg AUROC = {auc:.4f}")
    print(f"{'=' * 120}\n")


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== LR Classifier Evaluation on Math-Shepherd Stored Activations ===")
    print(f"Activations file:  {args.activations_file}")
    print(f"LR weights file:   {args.lr_weights_file}")
    print(f"LR metrics file:   {args.lr_metrics_file}")
    print(f"Target layers:     {args.target_layers}")
    print(f"Output dir:        {args.output_dir}")

    # ------------------------------------------------------------------
    # Load stored activations .pt (summary file — light)
    # ------------------------------------------------------------------
    print("\nLoading stored activations summary file...")
    stored   = torch.load(args.activations_file, map_location="cpu", weights_only=False)
    metadata = stored.get("metadata", {})
    layers_data = stored.get("layers", {})

    # Resolve raw activations directory
    raw_dir_str = args.raw_activations_dir or metadata.get("raw_activations_dir", None)
    if raw_dir_str is None:
        print("ERROR: raw_activations_dir not found in metadata and --raw_activations_dir not set.")
        sys.exit(1)
    raw_dir = Path(raw_dir_str)

    # Load shard index (lightweight — just counts and offsets)
    index      = torch.load(raw_dir / "index.pt", map_location="cpu", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]
    print(f"  Raw activations dir: {raw_dir}")
    print(f"  Shards: {num_shards}  |  d_model: {d_model}")

    # Per-sample fully-correct mask
    is_fully_correct = metadata["per_sample_is_fully_correct"].numpy().astype(bool)
    n_samples = len(is_fully_correct)
    print(f"  Total samples: {n_samples}  "
          f"(correct: {is_fully_correct.sum()}, flawed: {(~is_fully_correct).sum()})")

    # ------------------------------------------------------------------
    # Load LR weight vectors + intercepts
    # ------------------------------------------------------------------
    print("\nLoading LR weight vectors and intercepts...")
    lr_weights = torch.load(args.lr_weights_file, map_location="cpu", weights_only=False)

    intercepts: dict[tuple, float] = {}
    if args.lr_metrics_file and os.path.exists(args.lr_metrics_file):
        with open(args.lr_metrics_file, "r") as f:
            lr_metrics_json = json.load(f)
        for layer_str, layer_data in lr_metrics_json.items():
            if layer_str == "cross_weight_similarities":
                continue
            if isinstance(layer_data, dict):
                for gran, gran_data in layer_data.items():
                    if isinstance(gran_data, dict) and "intercept_input_space" in gran_data:
                        intercepts[(layer_str, gran)] = gran_data["intercept_input_space"]
        print(f"  Loaded {len(intercepts)} intercepts.")
    else:
        print("  WARNING: No lr_metrics_file found — intercepts default to 0.0.")

    # Index LR vectors by (layer_int, gran)
    lr_vectors: dict[tuple, torch.Tensor] = {}
    available_grans_set: set[str] = set()
    for layer_str, gran_dict in lr_weights.items():
        layer_int = int(layer_str)
        if layer_int not in args.target_layers:
            continue
        for gran, vec in gran_dict.items():
            lr_vectors[(layer_int, gran)] = vec.float()
            available_grans_set.add(gran)

    granularities   = sorted(args.granularities or available_grans_set)
    available_layers = sorted({l for l, _ in lr_vectors.keys()})
    print(f"  LR granularities:  {granularities}")
    print(f"  Available layers:  {available_layers}")

    def hook_name(layer: int) -> str:
        return f"blocks.{layer}.hook_out"

    def safe_intercept(layer: int, gran: str) -> float:
        return intercepts.get((str(layer), gran), 0.0)

    # ------------------------------------------------------------------
    # Build SAMPLE-LEVEL activations from the summary .pt
    # (per_sample_means already stored — no shard scanning needed)
    # ------------------------------------------------------------------
    print("\nBuilding sample-level activation dataset (from summary .pt)...")
    sample_acts: dict[int, np.ndarray] = {}   # layer -> (N, d_model) float32
    for layer in available_layers:
        hn  = hook_name(layer)
        psm = layers_data.get(hn, {}).get("per_sample_means", None)
        if psm is None:
            print(f"  WARNING: per_sample_means missing for {hn} — skipping layer {layer} for sample eval.")
            continue
        sample_acts[layer] = psm.float().numpy()

    sample_labels = is_fully_correct.astype(np.int32)   # (N,)
    print(f"  Sample-level: {len(sample_labels)} samples  "
          f"(correct: {sample_labels.sum()}, incorrect: {len(sample_labels)-sample_labels.sum()})")

    # ------------------------------------------------------------------
    # Build STEP-LEVEL activations via streaming shard aggregation
    # (mirrors run_step_level() — at most one shard in RAM at a time)
    # ------------------------------------------------------------------
    print("\nBuilding step-level activation dataset (streaming shards)...")
    step_data: dict[int, tuple] = {}   # layer -> (X, y, groups)

    for layer in available_layers:
        hn = hook_name(layer)
        shard_dir = raw_dir / hn.replace(".", "_")
        if not shard_dir.is_dir():
            print(f"  WARNING: shard dir not found for {hn} — skipping layer {layer} for step eval.")
            continue

        X, y, groups = aggregate_step_activations(raw_dir, hn, num_shards, d_model)
        step_data[layer] = (X, y, groups)

    # ------------------------------------------------------------------
    # Compute classifier scores & metrics
    # ------------------------------------------------------------------
    print("\nComputing classifier metrics...")

    eval_granularities = []
    if step_data:
        eval_granularities.append("step")
    if sample_acts:
        eval_granularities.append("sample")

    # clf_metrics[(lr_gran, layer, eval_gran)] -> metrics dict
    clf_metrics: dict[tuple, dict] = {}

    for layer in available_layers:
        for lr_gran in granularities:
            if (layer, lr_gran) not in lr_vectors:
                continue
            w_vec = lr_vectors[(layer, lr_gran)]   # (d_model,) float32
            b     = safe_intercept(layer, lr_gran) # scalar float

            # ---- Step-level ----
            if layer in step_data:
                X_step, y_step, groups_step = step_data[layer]
                scores = compute_scores(X_step, w_vec, b)
                m = compute_classifier_metrics(y_step, scores, threshold=0.0)
                m["n_groups"] = int(np.unique(groups_step).size)
                clf_metrics[(lr_gran, layer, "step")] = m

            # ---- Sample-level ----
            if layer in sample_acts:
                X_samp = sample_acts[layer]
                scores = compute_scores(X_samp, w_vec, b)
                m = compute_classifier_metrics(sample_labels, scores, threshold=0.0)
                clf_metrics[(lr_gran, layer, "sample")] = m

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    def safe_float(x):
        try:
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except Exception:
            return None

    json_out = {
        "config": {
            "activations_file":     args.activations_file,
            "lr_weights_file":      args.lr_weights_file,
            "lr_metrics_file":      args.lr_metrics_file,
            "target_layers":        args.target_layers,
            "granularities":        granularities,
            "eval_granularities":   eval_granularities,
            "n_samples_total":      n_samples,
            "n_samples_correct":    int(is_fully_correct.sum()),
            "n_samples_incorrect":  int((~is_fully_correct).sum()),
        },
        "classifier_metrics": {},
    }

    for (lr_gran, layer, eval_gran), m in clf_metrics.items():
        key = f"{lr_gran}__layer{layer}__eval_{eval_gran}"
        json_out["classifier_metrics"][key] = {
            k: (safe_float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in m.items()
        }

    results_path = os.path.join(args.output_dir,
                                "eval_results_classifier_math_shepherd.json")
    with open(results_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nClassifier results saved → {results_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(clf_metrics, available_layers, granularities, eval_granularities)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    subtitle = f"Math-Shepherd | LR grans: {', '.join(granularities)}"

    clf_plot_path = os.path.join(args.output_dir,
                                 "eval_plot_classifier_math_shepherd.png")
    plot_classifier_results(clf_metrics, available_layers, granularities,
                            eval_granularities, clf_plot_path, subtitle)

    cm_plot_path = os.path.join(args.output_dir,
                                "eval_plot_confusion_matrix_math_shepherd.png")
    plot_confusion_matrices(clf_metrics, available_layers, granularities,
                            eval_granularities, cm_plot_path)

    print("Done!")


if __name__ == "__main__":
    main()