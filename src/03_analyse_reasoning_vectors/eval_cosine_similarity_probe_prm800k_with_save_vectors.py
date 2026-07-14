"""
eval_cosine_similarity_probe_prm800k.py
========================================
Evaluates cosine-similarity-based linear probes built from the mean-difference
reasoning directions (reasoning_direction_step, reasoning_direction_sample).

Two evaluation modes
--------------------
1. K-FOLD CROSS-VALIDATION  (--mode cv)
   Fits the direction on K-1 folds of each input dataset separately, evaluates
   on the held-out fold.  Splitting is done at the **sample** level
   (StratifiedGroupKFold) so that steps from the same problem never appear in
   both train and test.

2. CROSS-DATASET HOLD-OUT  (--mode holdout)
   All ordered pairings between the input datasets are evaluated: for every
   ordered pair (A, B), the direction is fitted on A and evaluated on B.

Both modes evaluate all four probe / data combinations:
  - direction_step   -> classify steps
  - direction_step   -> classify samples    (cross-granularity)
  - direction_sample -> classify steps      (cross-granularity)
  - direction_sample -> classify samples

For each combination the classifier score is simply:
    score(a) = cosine_similarity(a, direction)
and the direction is recomputed (or re-sliced) from the TRAIN split only,
never from the evaluation data, ensuring no leakage.

Direction vector saving
-----------------------
For every input dataset and every layer, the script saves the full-dataset
reasoning direction vectors to:
    <output_dir>/<dataset_tag>/direction_vectors/layer_<N>/
        reasoning_direction_step_raw.pt          (raw float32, shape d_model)
        reasoning_direction_step_unit.pt         (unit-normalised, shape d_model)
        reasoning_direction_sample_raw.pt        (raw float32, shape d_model)
        reasoning_direction_sample_unit.pt       (unit-normalised, shape d_model)

Inputs
------
--datasets_vectors : one or more paths to reasoning_vectors_*.pt files.
                     Each entry must have a corresponding --datasets_raw entry.
--datasets_raw     : one or more paths to raw_activations directories.
                     Must be same length and order as --datasets_vectors.
--target_layers  : list of layer indices to evaluate  (default: 18-28)
--n_folds        : number of CV folds  (default: 5)
--mode           : "cv", "holdout", or "both"  (default: "holdout")
--output_dir     : where to write results JSON and direction vectors
--save_directions: (flag, default True) save direction vectors per dataset/layer

Backward-compatible legacy arguments are also accepted:
  --train_vectors / --train_raw / --eval_vectors / --eval_raw
  These are merged into --datasets_vectors / --datasets_raw before processing.

Usage examples
--------------
# CV only on a single dataset
python eval_cosine_similarity_probe_prm800k.py \\
    --mode cv \\
    --datasets_vectors .../processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \\
    --datasets_raw     .../processbench/raw_activations \\
    --output_dir    results/cosine_probe_cv

# Both CV and hold-out across three datasets (all ordered pairs for holdout)
python eval_cosine_similarity_probe_prm800k.py \\
    --mode both \\
    --datasets_vectors \\
        .../processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \\
        .../prm800k/reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt \\
        .../math-shepherd/reasoning_vectors_Qwen3-8B_math-shepherd_with_steps_avg_storage.pt \\
    --datasets_raw \\
        .../processbench/raw_activations \\
        .../prm800k/raw_activations \\
        .../math-shepherd/raw_activations \\
    --output_dir    results/cosine_probe_both
"""

import argparse
import gc
import json
import warnings
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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
    # ---- New unified multi-dataset arguments ----
    p.add_argument(
        "--datasets_vectors", type=str, nargs="+",
        default=[
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/math-shepherd/reasoning_vectors_Qwen3-8B_math-shepherd_with_steps_avg_storage.pt",
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt",
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt"
        ],
        help=(
            "One or more paths to reasoning_vectors .pt files. "
            "CV is run per dataset; holdout runs all ordered pairs. "
            "Must be the same length as --datasets_raw."
        ),
    )
    p.add_argument(
        "--datasets_raw", type=str, nargs="+",
        default=[
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/math-shepherd/raw_activations",
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations",
            "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations",
        ],
        help=(
            "One or more paths to raw_activations directories. "
            "Must be the same length and order as --datasets_vectors."
        ),
    )
    # ---- Legacy arguments (kept for backward compatibility) ----
    p.add_argument("--train_vectors", type=str, default=None,
                   help="[DEPRECATED] Use --datasets_vectors instead.")
    p.add_argument("--train_raw", type=str, default=None,
                   help="[DEPRECATED] Use --datasets_raw instead.")
    p.add_argument("--eval_vectors", type=str, nargs="+", default=None,
                   help="[DEPRECATED] Use --datasets_vectors instead.")
    p.add_argument("--eval_raw", type=str, nargs="+", default=None,
                   help="[DEPRECATED] Use --datasets_raw instead.")
    # ---- Shared arguments ----
    p.add_argument(
        "--target_layers", type=int, nargs="+",
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
        default="/home/ines/Reasoning-activations/results/cosine_probe_eval_layer_v2",
    )
    p.add_argument(
        "--save_directions", action="store_true", default=True,
        help=(
            "Save the full-dataset reasoning direction vectors (raw + unit-normalised) "
            "for each dataset and layer under <output_dir>/<dataset>/direction_vectors/layer_<N>/."
        ),
    )
    return p.parse_args()


# ============================================================
# Shard loading
# ============================================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    safe = hook_name.replace(".", "_")
    acts = torch.load(
        raw_dir / safe / f"shard_{shard_id:04d}.pt", weights_only=False
    ).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False)
    return acts, meta


# ============================================================
# PRM800K prefix cache loader
# ============================================================
def load_prefix_lookup(raw_dir: Path, hook_name: str, index: dict) -> dict:
    """Load all prefix shards -> dict: prefix_id -> (acts np.ndarray, meta list)."""
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
# Sample-level activation loader (from .pt vectors file)
# ============================================================
def load_sample_activations(vec_data: dict, hook_name: str, raw_dir: Path | None = None):
    """
    Read per-sample (or per-branch for PRM800K) mean activations from the vectors .pt file.

    Returns: sample_X (n, d_model), sample_y (n,), groups (n,)
    """
    per_sample_means      = vec_data["layers"][hook_name]["per_sample_means"]
    per_sample_is_correct = vec_data["metadata"]["per_sample_is_fully_correct"]
    n = min(len(per_sample_means), len(per_sample_is_correct))
    sample_X = per_sample_means[:n].to(torch.float32).numpy()
    sample_y = torch.tensor(per_sample_is_correct[:n]).to(torch.int32).numpy()

    groups = np.arange(n, dtype=np.int32)
    if raw_dir is not None:
        index_path = Path(raw_dir) / "index.pt"
        if index_path.exists():
            idx = torch.load(index_path, weights_only=False)
            has_prefix_dedup = bool(idx.get("prefix_index") and idx.get("num_prefix_shards", 0) > 0)
            if has_prefix_dedup:
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
# Step-level aggregation -- streams raw shards
# ============================================================
def aggregate_step_activations(raw_dir: Path, hook_name: str):
    """
    Stream all shards and return per-step mean activations.

    Returns: step_X (n_steps, d_model), step_y (n_steps,), step_groups (n_steps,)
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]
    has_prefix_dedup = bool(index.get("prefix_index") and index.get("num_prefix_shards", 0) > 0)

    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)

    if has_prefix_dedup:
        print(f"    Loading prefix shards (PRM800K dedup)...")
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
                key = (sample_idx, s_idx)
                step_sums[key]   += p_acts[row_i].astype(np.float64)
                step_counts[key] += 1
                if ic is False:
                    step_labels[key] = False
        del prefix_lookup
        gc.collect()

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
                key = (m["sample_idx"], b_idx, s_idx)
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
    step_groups = np.array([k[0] for k in step_keys], dtype=np.int32)
    del step_sums, step_counts, step_labels
    gc.collect()
    print(f"    Aggregated steps:  n={len(step_y)}  "
          f"(pos={step_y.sum()}  neg={len(step_y) - step_y.sum()})")
    return step_X, step_y, step_groups


# ============================================================
# Direction computation
# ============================================================
def compute_direction_raw(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    mean(X[y==1]) - mean(X[y==0]).
    Returns the RAW (unnormalised) direction vector, float32.
    """
    pos_mean = X[y == 1].mean(axis=0)
    neg_mean = X[y == 0].mean(axis=0)
    return (pos_mean - neg_mean).astype(np.float32)


def unit_normalise(v: np.ndarray) -> np.ndarray:
    """Return a unit-normalised copy of v (float32). Unchanged if near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def compute_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mean(X[y==1]) - mean(X[y==0]), unit-normalised."""
    return unit_normalise(compute_direction_raw(X, y))


def cosine_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of X and the (unit-norm) direction."""
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    return (X / row_norms) @ direction


def dot_product_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Raw dot product (no normalisation by row magnitudes)."""
    return X @ direction


# ============================================================
# Direction vector saving
# ============================================================
def save_direction_vectors(
    output_dir: Path,
    dataset_tag: str,
    layer: int,
    step_X: np.ndarray,
    step_y: np.ndarray,
    sample_X: np.ndarray,
    sample_y: np.ndarray,
):
    """
    Compute and save the full-dataset reasoning direction vectors (step and sample
    granularity) as .pt tensors, in both raw and unit-normalised forms.

    The step direction is the mean-difference vector computed from per-step
    averaged activations (step_X).  The sample direction uses per-sample
    averaged activations (sample_X).  Both are saved raw (float32) and as
    unit-normalised versions so callers can choose the form they need.

    Output layout:
        <output_dir>/<dataset_tag>/direction_vectors/layer_<layer>/
            reasoning_direction_step_raw.pt
            reasoning_direction_step_unit.pt
            reasoning_direction_sample_raw.pt
            reasoning_direction_sample_unit.pt
    """
    save_dir = output_dir / dataset_tag / "direction_vectors" / f"layer_{layer}"
    save_dir.mkdir(parents=True, exist_ok=True)

    def _compute_and_save(X, y, granularity):
        if len(np.unique(y)) < 2:
            print(f"    [save_directions] Skipping {granularity} direction for "
                  f"{dataset_tag}/layer_{layer}: single class in labels.")
            return
        raw  = compute_direction_raw(X, y)
        unit = unit_normalise(raw)
        for arr, fname in [
            (raw,  f"reasoning_direction_{granularity}_raw.pt"),
            (unit, f"reasoning_direction_{granularity}_unit.pt"),
        ]:
            path = save_dir / fname
            torch.save(torch.from_numpy(arr), path)
            print(f"    Saved {fname}  shape={arr.shape}  -> {path}")

    _compute_and_save(step_X,   step_y,   "step")
    _compute_and_save(sample_X, sample_y, "sample")


# ============================================================
# Metrics
# ============================================================
def compute_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate cosine-similarity scores as a binary classifier.
    Threshold is the midpoint between class means.
    """
    if len(np.unique(y)) < 2:
        return {"error": "single_class"}

    try:
        auroc = float(roc_auc_score(y, scores))
    except ValueError:
        auroc = float("nan")

    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    pooled_std = np.sqrt(
        (pos_scores.var() * len(pos_scores) + neg_scores.var() * len(neg_scores))
        / (len(pos_scores) + len(neg_scores) - 2 + 1e-12)
    )
    cohens_d = float((pos_scores.mean() - neg_scores.mean()) / (pooled_std + 1e-12))

    threshold = float((pos_scores.mean() + neg_scores.mean()) / 2.0)
    y_pred = (scores >= threshold).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1  = float(f1_score(y, y_pred, zero_division=0))

    tp = int(((y == 1) & (y_pred == 1)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    balanced_acc = float((sens + spec) / 2.0)

    denom = abs(float(neg_scores.mean()))
    selectivity = float(pos_scores.mean()) / denom if denom > 1e-12 else float("nan")

    try:
        _, mw_p = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        mw_p = float(mw_p)
    except Exception:
        mw_p = float("nan")

    return {
        "auroc":        auroc,
        "cohens_d":     cohens_d,
        "accuracy":     acc,
        "balanced_acc": balanced_acc,
        "f1":           f1,
        "selectivity":  selectivity,
        "mw_p":         mw_p,
        "threshold":    threshold,
        "mean_pos":     float(pos_scores.mean()),
        "mean_neg":     float(neg_scores.mean()),
        "std_pos":      float(pos_scores.std()),
        "std_neg":      float(neg_scores.std()),
        "n_pos":        int((y == 1).sum()),
        "n_neg":        int((y == 0).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ============================================================
# K-fold CV for one (direction_type, eval_granularity) pairing
# ============================================================
def run_cv_one_pairing(
    *,
    direction_X: np.ndarray,
    direction_y: np.ndarray,
    direction_groups: np.ndarray,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    eval_groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    """
    Group-stratified K-fold CV.

    Split is on eval_X/eval_groups; direction is refit from direction_X rows
    whose sample_idx falls in the training set.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(eval_X, eval_y, groups=eval_groups)
    ):
        train_sample_ids = set(eval_groups[train_idx].tolist())
        dir_mask = np.isin(direction_groups, list(train_sample_ids))
        if dir_mask.sum() < 2 or len(np.unique(direction_y[dir_mask])) < 2:
            print(f"      [{label}] fold {fold_idx}: insufficient training data -- skipping.")
            continue

        direction = compute_direction(direction_X[dir_mask], direction_y[dir_mask])

        fold_entry = {
            "fold":        fold_idx,
            "n_train_dir": int(dir_mask.sum()),
            "n_test_eval": int(len(test_idx)),
        }
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
            scorer_agg[f"{k}_mean"]  = float(np.mean(vals)) if vals else float("nan")
            scorer_agg[f"{k}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            scorer_agg[f"{k}_folds"] = vals
        agg[scorer_name] = scorer_agg
    for k in scalar_keys:
        agg[f"{k}_mean"] = agg["cosine"][f"{k}_mean"]
        agg[f"{k}_std"]  = agg["cosine"][f"{k}_std"]
    return agg


# ============================================================
# Hold-out evaluation (direction from full train set)
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
    result.update(result["cosine"])
    return result


# ============================================================
# Per-layer CV driver (single dataset)
# ============================================================
def evaluate_layer_cv(layer: int, train_data: dict, n_folds: int, seed: int) -> dict:
    """Run all four probe x granularity pairings (CV only) for one layer."""
    tr_sX = train_data["step_X"]
    tr_sy = train_data["step_y"]
    tr_sg = train_data["step_groups"]
    tr_mX = train_data["sample_X"]
    tr_my = train_data["sample_y"]
    tr_mg = train_data.get("sample_groups", np.arange(len(tr_my), dtype=np.int32))

    pairings = [
        ("step_dir->step_eval",     tr_sX, tr_sy, tr_sg, tr_sX, tr_sy, tr_sg),
        ("step_dir->sample_eval",   tr_sX, tr_sy, tr_sg, tr_mX, tr_my, tr_mg),
        ("sample_dir->step_eval",   tr_mX, tr_my, tr_mg, tr_sX, tr_sy, tr_sg),
        ("sample_dir->sample_eval", tr_mX, tr_my, tr_mg, tr_mX, tr_my, tr_mg),
    ]

    layer_results = {}
    for label, dX, dy, dg, eX, ey, eg in pairings:
        print(f"    [CV ] {label}")
        cv_res = run_cv_one_pairing(
            direction_X=dX, direction_y=dy, direction_groups=dg,
            eval_X=eX, eval_y=ey, eval_groups=eg,
            n_folds=n_folds, seed=seed, label=label,
        )
        layer_results[label] = {"cv": cv_res}
    return layer_results


# ============================================================
# Per-layer holdout driver (one ordered train->eval pair)
# ============================================================
def evaluate_layer_holdout(layer: int, train_data: dict, eval_data: dict) -> dict:
    """
    Run all four probe x granularity pairings (holdout only) for one layer.
    Direction is always fitted on the full train_data.
    """
    tr_sX = train_data["step_X"];   tr_sy = train_data["step_y"]
    tr_mX = train_data["sample_X"]; tr_my = train_data["sample_y"]
    ev_sX = eval_data["step_X"];    ev_sy = eval_data["step_y"]
    ev_mX = eval_data["sample_X"];  ev_my = eval_data["sample_y"]

    pairings = [
        ("step_dir->step_eval",     tr_sX, tr_sy, ev_sX, ev_sy),
        ("step_dir->sample_eval",   tr_sX, tr_sy, ev_mX, ev_my),
        ("sample_dir->step_eval",   tr_mX, tr_my, ev_sX, ev_sy),
        ("sample_dir->sample_eval", tr_mX, tr_my, ev_mX, ev_my),
    ]

    layer_results = {}
    for label, dX, dy, eX, ey in pairings:
        print(f"    [HO ] {label}")
        ho_res = run_holdout_one_pairing(
            direction_X=dX, direction_y=dy,
            eval_X=eX, eval_y=ey,
            label=label,
        )
        layer_results[label] = {"holdout": ho_res}
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
                    sr = r.get(sname, r)
                    print(
                        f"    [CV/{sname:6s}]  AUROC={sr['auroc_mean']:.4f}+-{sr['auroc_std']:.4f}  "
                        f"Cohen's d={sr['cohens_d_mean']:.4f}+-{sr['cohens_d_std']:.4f}  "
                        f"Bal.Acc={sr['balanced_acc_mean']:.4f}+-{sr['balanced_acc_std']:.4f}  "
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
# Dataset resolution (unified + legacy backward-compat)
# ============================================================
def _dataset_name_from_path(path: str) -> str:
    return Path(path).parent.name


def _resolve_datasets(args) -> tuple[list[str], list[str]]:
    """
    Merge new (--datasets_vectors/--datasets_raw) and legacy
    (--train_vectors/--train_raw + --eval_vectors/--eval_raw) arguments into
    a single unified list, de-duplicating by path.
    """
    vec_list = list(args.datasets_vectors) if args.datasets_vectors else []
    raw_list = list(args.datasets_raw)     if args.datasets_raw     else []

    # Prepend legacy --train_* if not already present
    if args.train_vectors and args.train_vectors not in vec_list:
        vec_list.insert(0, args.train_vectors)
        raw_list.insert(0, args.train_raw or "")

    # Append legacy --eval_* entries not already present
    if args.eval_vectors:
        ev_raw = args.eval_raw or []
        for vp, rp in zip(args.eval_vectors, ev_raw):
            if vp not in vec_list:
                vec_list.append(vp)
                raw_list.append(rp)

    if len(vec_list) != len(raw_list):
        raise ValueError(
            f"Number of vector paths ({len(vec_list)}) must match "
            f"number of raw-activation paths ({len(raw_list)})."
        )
    return vec_list, raw_list


def _unique_tags(paths: list[str]) -> list[str]:
    """Derive unique human-readable dataset tags from vector file paths."""
    raw_tags = [_dataset_name_from_path(p) for p in paths]
    seen: dict[str, int] = {}
    result = []
    for tag in raw_tags:
        if tag in seen:
            seen[tag] += 1
            result.append(f"{tag}_{seen[tag]}")
        else:
            seen[tag] = 0
            result.append(tag)
    return result


# ============================================================
# Summary tables
# ============================================================
def print_summary_tables(
    cv_results: dict,
    holdout_results: dict,
    dataset_tags: list[str],
    layers: list[int],
    do_cv: bool,
    do_holdout: bool,
):
    pairings = [
        "step_dir->step_eval",
        "step_dir->sample_eval",
        "sample_dir->step_eval",
        "sample_dir->sample_eval",
    ]

    def _table(title: str, get_data_fn, is_cv: bool):
        for scorer_name in ("cosine", "dot"):
            print(f"\n\n{'='*90}")
            print(f"  SUMMARY TABLE -- {title}  [{scorer_name}]")
            print(f"{'='*90}")
            header = f"{'Layer':>6}" + "".join(f"  {p:>26}" for p in pairings)
            print(header)
            print("-" * len(header))
            for layer in layers:
                row = f"{layer:>6}"
                for p in pairings:
                    pr = get_data_fn(layer, p)
                    sr = pr.get(scorer_name, pr)
                    if "error" in pr:
                        cell = "ERROR"
                    elif is_cv:
                        m = sr.get("auroc_mean", float("nan"))
                        s = sr.get("auroc_std",  float("nan"))
                        cell = f"{m:.4f}+-{s:.4f}"
                    else:
                        m = sr.get("auroc", float("nan"))
                        cell = f"{m:.4f}"
                    row += f"  {cell:>26}"
                print(row)

    if do_cv:
        for tag in dataset_tags:
            def _get_cv(layer, p, _tag=tag):
                return cv_results.get(_tag, {}).get(str(layer), {}).get(p, {}).get("cv", {})
            _table(f"CV on {tag} (mean AUROC +- std)", _get_cv, is_cv=True)

    if do_holdout:
        for (train_tag, eval_tag), ho_res in holdout_results.items():
            def _get_ho(layer, p, _hr=ho_res):
                return _hr.get(str(layer), {}).get(p, {}).get("holdout", {})
            _table(f"HOLD-OUT  train={train_tag}  eval={eval_tag}  (AUROC)", _get_ho, is_cv=False)


# ============================================================
# Entry point
# ============================================================
def main():
    args = parse_args()

    do_cv      = args.mode in ("cv", "both")
    do_holdout = args.mode in ("holdout", "both")

    vec_paths, raw_paths = _resolve_datasets(args)
    n_datasets   = len(vec_paths)
    dataset_tags = _unique_tags(vec_paths)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Datasets ({n_datasets}):")
    for i, (tag, vp, rp) in enumerate(zip(dataset_tags, vec_paths, raw_paths)):
        print(f"    [{i}] {tag}")
        print(f"        vectors : {vp}")
        print(f"        raw     : {rp}")
    print(f"  Mode             : {args.mode}")
    print(f"  Layers           : {args.target_layers}")
    if do_cv:
        print(f"  CV folds         : {args.n_folds}  (one run per dataset)")
    if do_holdout:
        n_pairs = n_datasets * (n_datasets - 1)
        print(f"  Holdout pairs    : {n_pairs}  (all ordered pairs among {n_datasets} datasets)")
    print(f"  Save directions  : {args.save_directions}")
    print(f"{'#'*70}\n")

    # Pre-load all vector files once
    print("Pre-loading all vector files ...")
    vec_cache: dict[str, dict] = {}
    for vp in vec_paths:
        if vp not in vec_cache:
            print(f"  Loading {vp}")
            vec_cache[vp] = torch.load(vp, weights_only=False, map_location="cpu")

    # Results containers
    # cv_results[dataset_tag][str(layer)][pairing] = {"cv": ...}
    cv_results: dict[str, dict] = {tag: {} for tag in dataset_tags}

    # holdout_results[(train_tag, eval_tag)][str(layer)][pairing] = {"holdout": ...}
    ho_pairs = list(permutations(range(n_datasets), 2)) if do_holdout else []
    holdout_results: dict[tuple, dict] = {
        (dataset_tags[i], dataset_tags[j]): {} for i, j in ho_pairs
    }

    # ------------------------------------------------------------------
    # Per-layer loop
    # ------------------------------------------------------------------
    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'#'*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'#'*70}")

        # Aggregate all datasets once for this layer
        dataset_data: dict[str, dict] = {}

        for ds_idx, (tag, vp, rp) in enumerate(zip(dataset_tags, vec_paths, raw_paths)):
            raw_dir = Path(rp)
            vec     = vec_cache[vp]

            print(f"\n  [{ds_idx}] Aggregating dataset: {tag}")
            print(f"    Step activations (shards) ...")
            sX, sy, sg = aggregate_step_activations(raw_dir, hook_name)
            print(f"    Sample activations (.pt) ...")
            mX, my, mg = load_sample_activations(vec, hook_name, raw_dir=raw_dir)

            dataset_data[tag] = {
                "step_X": sX, "step_y": sy, "step_groups": sg,
                "sample_X": mX, "sample_y": my, "sample_groups": mg,
            }

            # Save direction vectors for this dataset/layer
            if args.save_directions:
                save_direction_vectors(
                    output_dir=output_dir,
                    dataset_tag=tag,
                    layer=layer,
                    step_X=sX, step_y=sy,
                    sample_X=mX, sample_y=my,
                )

        # CV: one run per dataset
        if do_cv:
            print(f"\n  {'='*60}")
            print(f"  CV evaluation")
            print(f"  {'='*60}")
            for tag in dataset_tags:
                print(f"\n  CV on: {tag}")
                cv_layer = evaluate_layer_cv(
                    layer=layer,
                    train_data=dataset_data[tag],
                    n_folds=args.n_folds,
                    seed=args.seed,
                )
                cv_results[tag][str(layer)] = cv_layer
                print_layer_results(layer, cv_layer, do_cv=True, do_holdout=False)

        # Holdout: all ordered pairs
        if do_holdout:
            print(f"\n  {'='*60}")
            print(f"  Hold-out evaluation (all ordered pairs)")
            print(f"  {'='*60}")
            for (i, j) in ho_pairs:
                train_tag = dataset_tags[i]
                eval_tag  = dataset_tags[j]
                print(f"\n  Hold-out:  train={train_tag}  eval={eval_tag}")
                ho_layer = evaluate_layer_holdout(
                    layer=layer,
                    train_data=dataset_data[train_tag],
                    eval_data=dataset_data[eval_tag],
                )
                holdout_results[(train_tag, eval_tag)][str(layer)] = ho_layer
                print_layer_results(layer, ho_layer, do_cv=False, do_holdout=True)

        del dataset_data
        gc.collect()

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    if do_cv:
        for tag in dataset_tags:
            tag_dir = output_dir / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            out_path = tag_dir / "cosine_probe_cv_results.json"
            with open(out_path, "w") as f:
                json.dump(to_serialisable(cv_results[tag]), f, indent=2)
            print(f"\nCV results saved -> {out_path}")

    if do_holdout:
        for (train_tag, eval_tag), ho_res in holdout_results.items():
            ho_dir = output_dir / train_tag / f"eval_on_{eval_tag}"
            ho_dir.mkdir(parents=True, exist_ok=True)
            out_path = ho_dir / "cosine_probe_holdout_results.json"
            with open(out_path, "w") as f:
                json.dump(to_serialisable(ho_res), f, indent=2)
            print(f"Holdout results saved -> {out_path}")

    # Summary tables
    print_summary_tables(
        cv_results=cv_results,
        holdout_results=holdout_results,
        dataset_tags=dataset_tags,
        layers=args.target_layers,
        do_cv=do_cv,
        do_holdout=do_holdout,
    )


if __name__ == "__main__":
    main()