"""
eval_cosine_similarity_probe.py
================================
Evaluates cosine-similarity-based linear probes built from the mean-difference
reasoning directions (reasoning_direction_step, reasoning_direction_sample).

Two evaluation modes
--------------------
1. K-FOLD CROSS-VALIDATION  (--mode cv)
   Fits the direction on K-1 folds of one dataset, evaluates on the held-out
   fold.  Splitting is done at the **sample** level (StratifiedGroupKFold) so
   that steps from the same problem never appear in both train and test.

2. CROSS-DATASET HOLD-OUT  (--mode holdout)
   Fits the direction on the full SOURCE dataset, evaluates on the TARGET
   dataset.  No overlap possible by construction.

Both modes evaluate all four probe / data combinations:
  - direction_step   → classify steps
  - direction_step   → classify samples    (cross-granularity)
  - direction_sample → classify steps      (cross-granularity)
  - direction_sample → classify samples

For each combination the classifier score is simply:
    score(a) = cosine_similarity(a, direction)
and the direction is recomputed (or re-sliced) from the TRAIN split only,
never from the evaluation data, ensuring no leakage.

Inputs
------
--train_vectors  : path to reasoning_vectors_*.pt  (source / CV dataset)
--train_raw      : path to raw_activations dir      (source / CV dataset)
--eval_vectors   : one or more paths to reasoning_vectors_*.pt  (holdout targets)
--eval_raw       : one or more paths to raw_activations dirs    (same order as --eval_vectors)
--target_layers  : list of layer indices to evaluate  (default: 18-28)
--n_folds        : number of CV folds  (default: 5)
--mode           : "cv", "holdout", or "both"  (default: "both")
--output_dir     : where to write results JSON and plots

Usage examples
--------------
# CV only on ProcessBench
python eval_cosine_similarity_probe.py \\
    --mode cv \\
    --train_vectors .../processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \\
    --train_raw     .../processbench/raw_activations \\
    --output_dir    results/cosine_probe_cv

# Both CV and hold-out on two eval datasets
python eval_cosine_similarity_probe.py \\
    --mode both \\
    --train_vectors .../processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \\
    --train_raw     .../processbench/raw_activations \\
    --eval_vectors  .../prm800k/reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt \\
                    .../math-shepherd/reasoning_vectors_Qwen3-8B_math-shepherd_with_steps_avg_storage.pt \\
    --eval_raw      .../prm800k/raw_activations \\
                    .../math-shepherd/raw_activations \\
    --output_dir    results/cosine_probe_both
"""

import argparse
import gc
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Cosine-similarity probe evaluation for reasoning directions."
    )
    p.add_argument(
        "--train_vectors", type=str,
        default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt",
        help="Path to reasoning_vectors .pt file used for fitting / CV.",
    )
    p.add_argument(
        "--train_raw", type=str,
        default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations",
        help="Path to raw_activations directory for the train/CV dataset.",
    )
    p.add_argument(
        "--eval_vectors", type=str, nargs="+",
        default=[
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt"
        ],
        help=(
            "One or more paths to reasoning_vectors .pt files for hold-out evaluation. "
            "Each entry must have a corresponding --eval_raw entry at the same position."
        ),
    )
    p.add_argument(
        "--eval_raw", type=str, nargs="+",
        default=[
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations"
        ],
        help=(
            "One or more paths to raw_activations directories for hold-out evaluation. "
            "Must be the same length as --eval_vectors and correspond index-by-index."
        ),
    )
    p.add_argument(
        "--target_layers", type=int, nargs="+",
        #default=list(range(22, 24)), 
        default=list(range(18, 29)),
        help="Layer indices to evaluate.",
    )
    p.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode", type=str, default="holdout", choices=["cv", "holdout", "both"],
        help="Evaluation mode: 'cv', 'holdout', or 'both'.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="/home/ines/Reasoning-activations/results/cosine_probe_eval_layer",
    )
    return p.parse_args()

# ============================================================
# Shard loading  (same convention as train_lr_classifier)
# ============================================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    safe = hook_name.replace(".", "_")
    acts = torch.load(
        raw_dir / safe / f"shard_{shard_id:04d}.pt", weights_only=False
    ).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False)
    return acts, meta


def get_num_shards(raw_dir: Path, hook_name: str) -> int:
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    return index["num_shards"]

# ============================================================
# PRM800K prefix cache loader (mirrors train_lr_classifier_streaming_extended_prm800k.py)
# ============================================================
def load_prefix_lookup(raw_dir: Path, hook_name: str, index: dict) -> dict:
    """Load all prefix shards → dict: prefix_id → (acts np.ndarray, meta list)."""
    prefix_index      = index.get("prefix_index")
    num_prefix_shards = index.get("num_prefix_shards", 0)
    if not prefix_index or num_prefix_shards == 0:
        return {}

    safe        = hook_name.replace(".", "_")
    prefix_safe = "prefix_" + safe

    p_acts_list, p_meta_list = [], []
    for sid in range(num_prefix_shards):
        p_acts_list.append(
            torch.load(raw_dir / prefix_safe / f"shard_{sid:04d}.pt",
                       weights_only=False).to(torch.float32).numpy()
        )
        p_meta_list.extend(
            torch.load(raw_dir / "prefix_meta" / f"shard_{sid:04d}.pt",
                       weights_only=False)
        )

    prefix_acts_flat = np.concatenate(p_acts_list, axis=0)
    del p_acts_list

    prefix_lookup: dict[int, tuple] = {}
    for pid, (pstart, plen) in enumerate(prefix_index):
        prefix_lookup[pid] = (
            prefix_acts_flat[pstart: pstart + plen],
            p_meta_list[pstart: pstart + plen],
        )
    return prefix_lookup
# ============================================================
# Sample-level aggregation — reads directly from the .pt vectors file
# ============================================================
def load_sample_activations_old(vec_data: dict, hook_name: str):
    """
    Read pre-computed per-sample mean activations from the vectors .pt file.
    vec_data["layers"][hook_name]["per_sample_means"]   : (N_samples, d_model)
    vec_data["metadata"]["per_sample_is_fully_correct"] : (N_samples,) bool tensor
    Returns: sample_X (n, d_model) float32,  sample_y (n,) int32
    """
    per_sample_means = vec_data["layers"][hook_name]["per_sample_means"]
    per_sample_is_correct = vec_data["metadata"]["per_sample_is_fully_correct"]
    n = min(len(per_sample_means), len(per_sample_is_correct))
    sample_X = per_sample_means[:n].to(torch.float32).numpy()
    sample_y = per_sample_is_correct[:n].to(torch.int32).numpy()
    print(f"    Loaded samples from .pt:  n={n}  (pos={sample_y.sum()}  neg={n - sample_y.sum()})")
    return sample_X, sample_y

def load_sample_activations(vec_data: dict, hook_name: str, raw_dir: Path | None = None):
    """
    Read per-sample (or per-branch for PRM800K) mean activations from the vectors .pt file.

    For PRM800K (has_prefix_dedup=True):
      - per_sample_means : (N_branches, d_model)  — one row per (sample_idx, branch_idx)
      - per_sample_is_fully_correct : flat list/tensor over branches in sample_index order
      - groups returned : sample_idx per branch (for StratifiedGroupKFold leakage prevention)

    For other datasets:
      - per_sample_means : (N_samples, d_model)
      - groups returned : np.arange(n)  (each sample is its own group)
    """
    per_sample_means    = vec_data["layers"][hook_name]["per_sample_means"]
    per_sample_is_correct = vec_data["metadata"]["per_sample_is_fully_correct"]
    n = min(len(per_sample_means), len(per_sample_is_correct))
    sample_X = per_sample_means[:n].to(torch.float32).numpy()
    sample_y = torch.tensor(per_sample_is_correct[:n]).to(torch.int32).numpy()

    # Build sample-level groups for CV
    groups = np.arange(n, dtype=np.int32)  # default: each row is its own group
    if raw_dir is not None:
        index_path = Path(raw_dir) / "index.pt"
        if index_path.exists():
            idx = torch.load(index_path, weights_only=False)
            has_prefix_dedup = bool(idx.get("prefix_index") and idx.get("num_prefix_shards", 0) > 0)
            if has_prefix_dedup:
                # sample_index entries are 4-tuples: (start_row, num_branch_tokens, sample_idx, prefix_id)
                # Branches appear in dataset order; align with the n rows of per_sample_means.
                sample_index = idx["sample_index"]
                if len(sample_index) != len(per_sample_means):
                    print(f"    WARNING: sample_index length ({len(sample_index)}) != "
                        f"per_sample_means length ({len(per_sample_means)}). "
                        f"Using min={n} and hoping alignment holds.")
                branch_sample_ids = np.array([entry[2] for entry in sample_index[:n]], dtype=np.int32)
                groups = branch_sample_ids

    print(f"    Loaded samples from .pt:  n={n}  "
          f"(pos={sample_y.sum()}  neg={n - sample_y.sum()})  "
          f"n_groups={len(np.unique(groups))}")
    return sample_X, sample_y, groups

# ============================================================
# Step-level aggregation — streams raw shards
# ============================================================
def aggregate_step_activations_old(raw_dir: Path, hook_name: str):
    """
    Stream all shards and return per-step mean activations.
    Returns: step_X (n_steps, d_model) float32,
             step_y (n_steps,) int32,
             step_groups (n_steps,) int32  [sample_idx, for GroupKFold]
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model = index["d_model"]

    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)

    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)
        for i, m in enumerate(meta):
            if m.get("step_idx", -1) < 0 or m.get("is_correct") is None:
                continue
            key = (m["sample_idx"], m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if m["is_correct"] is False:
                step_labels[key] = False
        del acts, acts_np, meta
        gc.collect()

    step_keys   = sorted(step_sums.keys())
    step_X      = np.stack([step_sums[k] / step_counts[k] for k in step_keys]).astype(np.float32)
    step_y      = np.array([1 if step_labels[k] else 0 for k in step_keys], dtype=np.int32)
    step_groups = np.array([k[0] for k in step_keys], dtype=np.int32)
    del step_sums, step_counts, step_labels
    gc.collect()
    print(f"    Aggregated steps from shards:  n={len(step_y)}  (pos={step_y.sum()}  neg={len(step_y) - step_y.sum()})")
    return step_X, step_y, step_groups

def aggregate_step_activations(raw_dir: Path, hook_name: str):
    """
    Stream all shards and return per-step mean activations.

    For PRM800K (has_prefix_dedup):
      - Prefix steps: keyed (sample_idx, step_idx), loaded once from prefix shards.
      - Branch steps: keyed (sample_idx, branch_idx, step_idx), one per branch.
      - groups = sample_idx (key[0]) so steps from the same problem stay together in CV.

    For other datasets: key = (sample_idx, step_idx), unchanged.
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]
    has_prefix_dedup = bool(index.get("prefix_index") and index.get("num_prefix_shards", 0) > 0)

    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)

    # ---- PRM800K: seed with prefix steps first (each prefix loaded once) ----
    if has_prefix_dedup:
        print(f"    Loading prefix shards for step aggregation (PRM800K dedup)...")
        prefix_lookup = load_prefix_lookup(raw_dir, hook_name, index)
        for pid, (p_acts, p_meta) in prefix_lookup.items():
            if not p_meta:
                continue
            sample_idx = p_meta[0]["sample_idx"]
            for row_i, m in enumerate(p_meta):
                s_idx = m.get("step_idx", -1)
                ic    = m.get("is_correct")
                if s_idx < 0 or ic is None:
                    continue
                key = (sample_idx, s_idx)        # prefix step: no branch_idx
                step_sums[key]   += p_acts[row_i].astype(np.float64)
                step_counts[key] += 1
                if ic is False:
                    step_labels[key] = False
        del prefix_lookup; gc.collect()

    # ---- Stream branch (or full-sequence) shards ----
    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)
        for i, m in enumerate(meta):
            s_idx = m.get("step_idx", -1)
            ic    = m.get("is_correct")
            if s_idx < 0 or ic is None:
                continue
            if has_prefix_dedup:
                b_idx = m.get("branch_idx", 0)
                key = (m["sample_idx"], b_idx, s_idx)   # branch step: include branch_idx
            else:
                key = (m["sample_idx"], s_idx)
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if ic is False:
                step_labels[key] = False
        del acts, acts_np, meta
        gc.collect()

    step_keys   = sorted(step_sums.keys())
    step_X      = np.stack([step_sums[k] / step_counts[k] for k in step_keys]).astype(np.float32)
    step_y      = np.array([1 if step_labels[k] else 0 for k in step_keys], dtype=np.int32)
    step_groups = np.array([k[0] for k in step_keys], dtype=np.int32)   # always sample_idx
    del step_sums, step_counts, step_labels
    gc.collect()
    print(f"    Aggregated steps:  n={len(step_y)}  (pos={step_y.sum()}  neg={len(step_y)-step_y.sum()})")
    return step_X, step_y, step_groups

# (kept for reference — no longer called)
def aggregate_activations(raw_dir: Path, hook_name: str, per_sample_is_correct: torch.Tensor):
    """
    Stream all shards and return:
      step_X      : (n_steps, d_model)  float32  — mean activation per step
      step_y      : (n_steps,)          int32    — 1=correct 0=incorrect
      step_groups : (n_steps,)          int32    — sample_idx (for GroupKFold)
      sample_X    : (n_samples, d_model) float32 — mean activation per sample
      sample_y    : (n_samples,)         int32   — 1=fully correct 0=flawed
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]

    d_model = index["d_model"]

    # Step-level accumulators: keyed by (sample_idx, step_idx)
    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)   # True = correct until proven wrong

    # Sample-level accumulators: keyed by sample_idx
    sample_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    sample_counts = defaultdict(int)

    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)

        for i, m in enumerate(meta):
            s_idx = m["sample_idx"]

            # ---- sample level (all tokens regardless of step label) ----
            sample_sums[s_idx]   += acts_np[i]
            sample_counts[s_idx] += 1

            # ---- step level (only labelled tokens) ----
            if m.get("step_idx", -1) < 0 or m.get("is_correct") is None:
                continue
            key = (s_idx, m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if m["is_correct"] is False:
                step_labels[key] = False

        del acts, acts_np, meta
        gc.collect()

    # ---------- Build step arrays ----------
    step_keys = sorted(step_sums.keys())
    step_X = np.stack(
        [step_sums[k] / step_counts[k] for k in step_keys]
    ).astype(np.float32)
    step_y = np.array([1 if step_labels[k] else 0 for k in step_keys], dtype=np.int32)
    step_groups = np.array([k[0] for k in step_keys], dtype=np.int32)

    # ---------- Build sample arrays ----------
    sample_ids = sorted(sample_sums.keys())
    # Clip to length of label tensor (handles off-by-one in some datasets)
    n_samples = min(len(sample_ids), len(per_sample_is_correct))
    sample_ids = sample_ids[:n_samples]
    sample_X = np.stack(
        [sample_sums[s] / sample_counts[s] for s in sample_ids]
    ).astype(np.float32)
    sample_y = np.array(
        [int(per_sample_is_correct[s].item()) for s in sample_ids], dtype=np.int32
    )

    del step_sums, step_counts, step_labels, sample_sums, sample_counts
    gc.collect()

    print(
        f"    Aggregated  steps: {len(step_y)}  "
        f"(pos={step_y.sum()} neg={len(step_y)-step_y.sum()})  |  "
        f"samples: {len(sample_y)}  "
        f"(pos={sample_y.sum()} neg={len(sample_y)-sample_y.sum()})"
    )
    return step_X, step_y, step_groups, sample_X, sample_y


# ============================================================
# Direction computation
# ============================================================
def compute_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    mean(X[y==1]) - mean(X[y==0])
    Returns a unit-normalised direction vector of shape (d_model,).
    """
    pos_mean = X[y == 1].mean(axis=0)
    neg_mean = X[y == 0].mean(axis=0)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return direction  # degenerate; caller should handle
    return direction / norm


def cosine_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between each row of X and the (already unit-norm) direction.
    """
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    return (X / row_norms) @ direction  # (N,)


def dot_product_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Raw dot product between each row of X and the (unit-norm) direction.
    No normalisation by row magnitudes.
    """
    return X @ direction  # (N,)


# ============================================================
# Metrics
# ============================================================
def compute_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate cosine-similarity scores as a binary classifier.
    Threshold is chosen as the midpoint between class means (sign-optimal for
    Gaussian-equal-variance distributions; no free parameter).
    """
    if len(np.unique(y)) < 2:
        return {"error": "single_class"}

    # AUROC — threshold-free
    try:
        auroc = float(roc_auc_score(y, scores))
    except ValueError:
        auroc = float("nan")

    # Cohen's d (effect size)
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    pooled_std = np.sqrt(
        (pos_scores.var() * len(pos_scores) + neg_scores.var() * len(neg_scores))
        / (len(pos_scores) + len(neg_scores) - 2 + 1e-12)
    )
    cohens_d = float((pos_scores.mean() - neg_scores.mean()) / (pooled_std + 1e-12))

    # Threshold = midpoint of class means (maximises balanced accuracy for equal-variance Gaussians)
    threshold = float((pos_scores.mean() + neg_scores.mean()) / 2.0)
    y_pred = (scores >= threshold).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1  = float(f1_score(y, y_pred, zero_division=0))

    # Balanced accuracy (= average recall across classes)
    tp = int(((y == 1) & (y_pred == 1)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    balanced_acc = float((sens + spec) / 2.0)

    # Selectivity ratio: mean positive score / mean negative score
    # (Guard against negative or zero denominators)
    denom = abs(float(neg_scores.mean()))
    selectivity = float(pos_scores.mean()) / denom if denom > 1e-12 else float("nan")

    # Mann-Whitney U (non-parametric significance)
    try:
        _, mw_p = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        mw_p = float(mw_p)
    except Exception:
        mw_p = float("nan")

    return {
        "auroc":         auroc,
        "cohens_d":      cohens_d,
        "accuracy":      acc,
        "balanced_acc":  balanced_acc,
        "f1":            f1,
        "selectivity":   selectivity,
        "mw_p":          mw_p,
        "threshold":     threshold,
        "mean_pos":      float(pos_scores.mean()),
        "mean_neg":      float(neg_scores.mean()),
        "std_pos":       float(pos_scores.std()),
        "std_neg":       float(neg_scores.std()),
        "n_pos":         int((y == 1).sum()),
        "n_neg":         int((y == 0).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ============================================================
# K-fold CV for one (direction_type, eval_granularity) pairing
# ============================================================
def run_cv_one_pairing(
    *,
    direction_X: np.ndarray,  # activations used to compute the direction
    direction_y: np.ndarray,
    direction_groups: np.ndarray,  # sample_idx for direction_X rows (step-level)
    eval_X: np.ndarray,            # activations being classified
    eval_y: np.ndarray,
    eval_groups: np.ndarray,       # sample_idx for eval_X rows
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    """
    Run group-stratified K-fold CV.

    The split is done on `eval_X` (what we want to classify) using
    `eval_groups` to prevent cross-sample leakage.

    For each fold:
      - Identify the train sample_idx set from eval split
      - Compute direction from `direction_X` rows whose sample_idx is in the
        train set (i.e., only if direction and eval share the same dataset,
        which is always true in CV mode)
      - Evaluate on held-out eval rows

    If direction_granularity != eval_granularity (cross-granularity),
    direction_groups and eval_groups are still both sample_idx arrays, so
    the same train/test sample split applies to both.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(eval_X, eval_y, groups=eval_groups)
    ):
        train_sample_ids = set(eval_groups[train_idx].tolist())

        # Recompute direction using only training samples
        dir_mask = np.isin(direction_groups, list(train_sample_ids))
        if dir_mask.sum() < 2 or len(np.unique(direction_y[dir_mask])) < 2:
            print(f"      [{label}] fold {fold_idx}: insufficient training data — skipping.")
            continue

        direction = compute_direction(direction_X[dir_mask], direction_y[dir_mask])

        # Score held-out eval rows with both scorers
        fold_entry = {"fold": fold_idx, "n_train_dir": int(dir_mask.sum()), "n_test_eval": int(len(test_idx))}
        for scorer_name, scorer_fn in [("cosine", cosine_scores), ("dot", dot_product_scores)]:
            scores_test = scorer_fn(eval_X[test_idx], direction)
            fold_entry[scorer_name] = compute_metrics(scores_test, eval_y[test_idx])
        fold_metrics.append(fold_entry)

    if not fold_metrics:
        return {"error": "no_valid_folds"}

    scalar_keys = ["auroc", "cohens_d", "accuracy", "balanced_acc", "f1", "selectivity", "mw_p"]
    agg = {"folds": fold_metrics, "n_folds": len(fold_metrics)}
    for scorer_name in ("cosine", "dot"):
        scorer_agg = {}
        for k in scalar_keys:
            vals = [fm[scorer_name][k] for fm in fold_metrics
                    if isinstance(fm[scorer_name].get(k), float) and np.isfinite(fm[scorer_name][k])]
            scorer_agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            scorer_agg[f"{k}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            scorer_agg[f"{k}_folds"] = vals
        agg[scorer_name] = scorer_agg
    # Back-compat: top-level keys mirror cosine scorer
    for k in scalar_keys:
        agg[f"{k}_mean"] = agg["cosine"][f"{k}_mean"]
        agg[f"{k}_std"]  = agg["cosine"][f"{k}_std"]
    return agg


# ============================================================
# Hold-out evaluation (direction from full train, eval on target)
# ============================================================
def run_holdout_one_pairing(
    *,
    direction_X: np.ndarray,
    direction_y: np.ndarray,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    label: str,
) -> dict:
    if len(np.unique(direction_y)) < 2:
        return {"error": "single_class_direction"}
    if len(np.unique(eval_y)) < 2:
        return {"error": "single_class_eval"}

    direction = compute_direction(direction_X, direction_y)
    result = {"n_train_dir": len(direction_X), "n_eval": len(eval_X)}
    for scorer_name, scorer_fn in [("cosine", cosine_scores), ("dot", dot_product_scores)]:
        scores = scorer_fn(eval_X, direction)
        result[scorer_name] = compute_metrics(scores, eval_y)
    # Back-compat: top-level keys mirror cosine scorer
    result.update(result["cosine"])
    return result


# ============================================================
# Main per-layer evaluation driver
# ============================================================
def evaluate_layer(
    layer: int,
    train_data: dict,  # {"step_X", "step_y", "step_groups", "sample_X", "sample_y"}
    eval_data: dict | None,  # same keys; None if holdout not requested
    n_folds: int,
    seed: int,
    do_cv: bool,
    do_holdout: bool,
) -> dict:
    """
    Run all four probe × granularity pairings for one layer.

    Pairings:
      step_dir  → step_eval    (direction from steps,  classify steps)
      step_dir  → sample_eval  (direction from steps,  classify samples)
      sample_dir → step_eval   (direction from samples, classify steps)
      sample_dir → sample_eval (direction from samples, classify samples)
    """
    layer_results = {}

    # Convenience aliases for train data
    tr_sX = train_data["step_X"]
    tr_sy = train_data["step_y"]
    tr_sg = train_data["step_groups"]   # sample_idx per step row
    tr_mX = train_data["sample_X"]
    tr_my = train_data["sample_y"]
    # For samples, the "group" is the sample itself
    # tr_mg = np.arange(len(tr_my), dtype=np.int32)
    # tr_mg = train_data["sample_groups"]
    tr_mg = train_data.get("sample_groups", np.arange(len(tr_my), dtype=np.int32))

    pairings = [
        # label,  direction_X, direction_y, direction_groups, eval_X, eval_y, eval_groups
        ("step_dir→step_eval",   tr_sX, tr_sy, tr_sg, tr_sX, tr_sy, tr_sg),
        ("step_dir→sample_eval", tr_sX, tr_sy, tr_sg, tr_mX, tr_my, tr_mg),
        ("sample_dir→step_eval", tr_mX, tr_my, tr_mg, tr_sX, tr_sy, tr_sg),
        ("sample_dir→sample_eval", tr_mX, tr_my, tr_mg, tr_mX, tr_my, tr_mg),
    ]

    for label, dX, dy, dg, eX, ey, eg in pairings:
        layer_results[label] = {}

        if do_cv:
            print(f"    [CV ] {label}")
            cv_res = run_cv_one_pairing(
                direction_X=dX, direction_y=dy, direction_groups=dg,
                eval_X=eX, eval_y=ey, eval_groups=eg,
                n_folds=n_folds, seed=seed, label=label,
            )
            layer_results[label]["cv"] = cv_res

        if do_holdout and eval_data is not None:
            ev_sX = eval_data["step_X"]
            ev_sy = eval_data["step_y"]
            ev_mX = eval_data["sample_X"]
            ev_my = eval_data["sample_y"]

            # For hold-out: direction is always from FULL train set
            # eval target depends on the pairing label
            if "→step_eval" in label:
                ho_eX, ho_ey = ev_sX, ev_sy
            else:
                ho_eX, ho_ey = ev_mX, ev_my

            print(f"    [HO ] {label}")
            ho_res = run_holdout_one_pairing(
                direction_X=dX, direction_y=dy,
                eval_X=ho_eX, eval_y=ho_ey,
                label=label,
            )
            layer_results[label]["holdout"] = ho_res

    return layer_results


# ============================================================
# Pretty printer
# ============================================================
def print_layer_results(layer: int, layer_results: dict, do_cv: bool, do_holdout: bool):
    print(f"\n{'='*70}")
    print(f"  LAYER {layer}")
    print(f"{'='*70}")
    for pairing, modes in layer_results.items():
        print(f"\n  {pairing}")
        if do_cv and "cv" in modes:
            r = modes["cv"]
            if "error" in r:
                print(f"    [CV]  ERROR: {r['error']}")
            else:
                for sname in ("cosine", "dot"):
                    sr = r.get(sname, r)  # fallback to r for back-compat
                    print(
                        f"    [CV/{sname:6s}]  AUROC={sr['auroc_mean']:.4f}±{sr['auroc_std']:.4f}  "
                        f"Cohen's d={sr['cohens_d_mean']:.4f}±{sr['cohens_d_std']:.4f}  "
                        f"Bal.Acc={sr['balanced_acc_mean']:.4f}±{sr['balanced_acc_std']:.4f}  "
                        f"(n_folds={r['n_folds']})"
                    )
        if do_holdout and "holdout" in modes:
            r = modes["holdout"]
            if "error" in r:
                print(f"    [HO]  ERROR: {r['error']}")
            else:
                for sname in ("cosine", "dot"):
                    sr = r.get(sname, r)
                    print(
                        f"    [HO/{sname:6s}]  AUROC={sr['auroc']:.4f}  "
                        f"Cohen's d={sr['cohens_d']:.4f}  "
                        f"Bal.Acc={sr['balanced_acc']:.4f}  "
                        f"MW-p={sr['mw_p']:.3e}  "
                        f"n_eval={r.get('n_eval','?')}"
                    )


# ============================================================
# JSON serialisation helper
# ============================================================
def to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# Entry point
# ============================================================
def _dataset_name_from_path(path: str) -> str:
    """Extract dataset tag from the vectors file path (the parent directory name)."""
    return Path(path).parent.name


def main():
    args = parse_args()

    do_cv      = args.mode in ("cv", "both")
    do_holdout = args.mode in ("holdout", "both")

    # ------------------------------------------------------------------
    # Normalise eval_vectors / eval_raw to lists (nargs="+" always gives
    # a list, but guard against accidental single-string defaults).
    # ------------------------------------------------------------------
    eval_vectors_list = args.eval_vectors if isinstance(args.eval_vectors, list) else [args.eval_vectors]
    eval_raw_list     = args.eval_raw     if isinstance(args.eval_raw,     list) else [args.eval_raw]

    if do_holdout and len(eval_vectors_list) != len(eval_raw_list):
        raise ValueError(
            f"--eval_vectors ({len(eval_vectors_list)} entries) and "
            f"--eval_raw ({len(eval_raw_list)} entries) must have the same number of paths."
        )

    eval_datasets = list(zip(eval_vectors_list, eval_raw_list))  # [(vec_path, raw_dir), ...]

    train_tag = _dataset_name_from_path(args.train_vectors)

    # .../cosine_probe_eval/processbench/
    cv_dir = Path(args.output_dir) / train_tag
    cv_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load train vector file ----------
    print(f"\nLoading train vectors file: {args.train_vectors}")
    train_vec     = torch.load(args.train_vectors, weights_only=False, map_location="cpu")
    train_raw_dir = Path(args.train_raw)

    # ---------- Pre-load all eval vector files ----------
    eval_vec_cache = {}   # path → loaded dict
    if do_holdout:
        for ev_vec_path, _ in eval_datasets:
            if ev_vec_path not in eval_vec_cache:
                print(f"Loading eval vectors file: {ev_vec_path}")
                eval_vec_cache[ev_vec_path] = torch.load(
                    ev_vec_path, weights_only=False, map_location="cpu"
                )

    # ------------------------------------------------------------------
    # Per-layer loop
    # ------------------------------------------------------------------
    # cv_results      : layer → pairing → cv metrics  (single, from train)
    # holdout_results : eval_tag → layer → pairing → holdout metrics
    # all_results     : layer → pairing → {cv, holdout_<tag>, ...}
    # ------------------------------------------------------------------
    cv_results      = {}
    holdout_results = {_dataset_name_from_path(ev): {} for ev, _ in eval_datasets} if do_holdout else {}
    all_results     = {}

    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'#'*70}")
        print(f"  Processing layer {layer}  ({hook_name})")
        print(f"{'#'*70}")

        # ---------- Aggregate TRAIN dataset ----------
        print("  Aggregating TRAIN step activations (shards) …")
        tr_sX, tr_sy, tr_sg = aggregate_step_activations(train_raw_dir, hook_name)
        print("  Loading TRAIN sample activations (from .pt) …")
        tr_mX, tr_my, tr_mg = load_sample_activations(train_vec, hook_name, raw_dir=train_raw_dir)
        train_data = {
            "step_X": tr_sX, "step_y": tr_sy, "step_groups": tr_sg,
            "sample_X": tr_mX, "sample_y": tr_my, "sample_groups": tr_mg,
        }

        # ---------- CV (train-only, run once) ----------
        layer_results = {}
        if do_cv:
            cv_layer = evaluate_layer(
                layer=layer,
                train_data=train_data,
                eval_data=None,
                n_folds=args.n_folds,
                seed=args.seed,
                do_cv=True,
                do_holdout=False,
            )
            cv_results[str(layer)] = {
                p: {"cv": v["cv"]} for p, v in cv_layer.items() if "cv" in v
            }
            for p, v in cv_layer.items():
                layer_results.setdefault(p, {})["cv"] = v["cv"]
            print_layer_results(layer, cv_layer, do_cv=True, do_holdout=False)

        # ---------- Hold-out: iterate over each eval dataset ----------
        if do_holdout:
            for ev_vec_path, ev_raw_path in eval_datasets:
                eval_tag     = _dataset_name_from_path(ev_vec_path)
                eval_raw_dir = Path(ev_raw_path)
                eval_vec     = eval_vec_cache[ev_vec_path]

                print(f"\n  --- Hold-out on: {eval_tag} ---")
                print(f"  Aggregating EVAL step activations …")
                ev_sX, ev_sy, ev_sg = aggregate_step_activations(eval_raw_dir, hook_name)
                print(f"  Loading EVAL sample activations …")
                ev_mX, ev_my, ev_mg = load_sample_activations(eval_vec, hook_name, raw_dir=eval_raw_dir)
                eval_data = {
                    "step_X": ev_sX, "step_y": ev_sy, "step_groups": ev_sg,
                    "sample_X": ev_mX, "sample_y": ev_my, "sample_groups": ev_mg,
                }

                ho_layer = evaluate_layer(
                    layer=layer,
                    train_data=train_data,
                    eval_data=eval_data,
                    n_folds=args.n_folds,
                    seed=args.seed,
                    do_cv=False,
                    do_holdout=True,
                )

                holdout_results[eval_tag][str(layer)] = {
                    p: {"holdout": v["holdout"]} for p, v in ho_layer.items() if "holdout" in v
                }
                for p, v in ho_layer.items():
                    layer_results.setdefault(p, {})[f"holdout_{eval_tag}"] = v.get("holdout", {})

                print_layer_results(layer, ho_layer, do_cv=False, do_holdout=True)

                del ev_sX, ev_sy, ev_sg, ev_mX, ev_my, ev_mg, eval_data
                gc.collect()

        all_results[str(layer)] = layer_results

        # Free train memory before next layer
        del tr_sX, tr_sy, tr_sg, tr_mX, tr_my, tr_mg, train_data
        gc.collect()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if do_cv and cv_results:
        out_path = cv_dir / "cosine_probe_cv_results.json"
        with open(out_path, "w") as f:
            json.dump(to_serialisable(cv_results), f, indent=2)
        print(f"\nCV results saved → {out_path}")

    if do_holdout:
        for eval_tag, ho_res in holdout_results.items():
            holdout_dir = cv_dir / f"eval_on_{eval_tag}"
            holdout_dir.mkdir(parents=True, exist_ok=True)
            out_path = holdout_dir / "cosine_probe_holdout_results.json"
            with open(out_path, "w") as f:
                json.dump(to_serialisable(ho_res), f, indent=2)
            print(f"Holdout results saved → {out_path}")

    # ---------- Print summary tables ----------
    print_summary_table(all_results, do_cv, do_holdout, args.target_layers,
                        eval_tags=[_dataset_name_from_path(ev) for ev, _ in eval_datasets] if do_holdout else [])


# ============================================================
# Summary table across layers
# ============================================================
def print_summary_table(results: dict, do_cv: bool, do_holdout: bool, layers: list,
                        eval_tags: list | None = None):
    pairings = [
        "step_dir→step_eval",
        "step_dir→sample_eval",
        "sample_dir→step_eval",
        "sample_dir→sample_eval",
    ]

    eval_tags = eval_tags or []

    modes_to_print = []
    if do_cv:
        modes_to_print.append(("CV (mean AUROC ± std)", "cv"))
    if do_holdout:
        for tag in eval_tags:
            modes_to_print.append((f"HOLD-OUT on {tag} (AUROC)", f"holdout_{tag}"))

    for mode_label, mode_key in modes_to_print:
        is_cv = mode_key == "cv"
        for scorer_name in ("cosine", "dot"):
            print(f"\n\n{'='*90}")
            print(f"  SUMMARY TABLE — {mode_label}  [{scorer_name}]")
            print(f"{'='*90}")
            header = f"{'Layer':>6}" + "".join(f"  {p:>24}" for p in pairings)
            print(header)
            print("-" * len(header))
            for layer in layers:
                lr = results.get(str(layer), {})
                row = f"{layer:>6}"
                for p in pairings:
                    pr = lr.get(p, {}).get(mode_key, {})
                    sr = pr.get(scorer_name, pr)  # fallback for back-compat
                    if "error" in pr:
                        cell = "ERROR"
                    elif is_cv:
                        m, s = sr.get("auroc_mean", float("nan")), sr.get("auroc_std", float("nan"))
                        cell = f"{m:.4f}±{s:.4f}"
                    else:
                        m = sr.get("auroc", float("nan"))
                        cell = f"{m:.4f}"
                    row += f"  {cell:>24}"
                print(row)


if __name__ == "__main__":
    main()