"""
eval_cosine_similarity_probe_prm800k_with_deconfounding.py
===========================================================
Extends eval_cosine_similarity_probe_prm800k_with_save_vectors.py with
optional PCA-based deconfounding of reasoning directions.

All existing functionality is preserved.  When one or more PCA component
files are supplied via --pca_files, every analysis is run twice:
  (a) on the original (raw) step / sample activations, and
  (b) on PCA-deconfounded versions of those activations,
      once per PCA file supplied.

PCA component file format (produced by run_fw_pass_with_step_averaging_storage_v2.py
in --type baseline mode):
    {
        "layers": {
            <int layer_idx>: Tensor[n_components, d_model],   # e.g. key=18 → float32
            ...
        },
        "metadata": { ... }   # informational only
    }

Deconfounding logic (identical to deconfounding_step_reasoning_vectors.ipynb):
    For each PCA component c_i (a unit-norm row vector of shape [d_model]):
        x ← x - (x · c_i) * c_i
    applied row-by-row to every activation matrix, and to every direction
    vector before scoring.

New CLI argument
----------------
--pca_files : zero or more paths to PCA .pt files.
              Each produces an independent deconfounded analysis labelled
              by the stem of its filename (e.g. "fineweb_pca_components").
              If not supplied (default), only the original analysis is run.

Output structure (additions only — existing outputs are unchanged)
------------------------------------------------------------------
For each PCA file <pca_tag>:
  Direction vectors:
    <output_dir>/<dataset_tag>/direction_vectors/layer_<N>/
        reasoning_direction_step_raw_deconf_<pca_tag>.pt
        reasoning_direction_step_unit_deconf_<pca_tag>.pt
        reasoning_direction_sample_raw_deconf_<pca_tag>.pt
        reasoning_direction_sample_unit_deconf_<pca_tag>.pt

  JSON results (CV):
    <output_dir>/<dataset_tag>/cosine_probe_cv_results_deconf_<pca_tag>.json
  JSON results (holdout):
    <output_dir>/<train_tag>/eval_on_<eval_tag>/cosine_probe_holdout_results_deconf_<pca_tag>.json

Summary tables are printed for both original and deconfounded versions.

Usage examples
--------------
# Existing usage (no change):
python eval_cosine_similarity_probe_prm800k_with_deconfounding.py \\
    --mode holdout \\
    --datasets_vectors .../processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \\
    --datasets_raw     .../processbench/raw_activations \\
    --output_dir       results/cosine_probe_eval

# With deconfounding (one PCA file):
python eval_cosine_similarity_probe_prm800k_with_deconfounding.py \\
    --mode both \\
    --datasets_vectors .../processbench/... .../prm800k/... \\
    --datasets_raw     .../processbench/raw_activations .../prm800k/raw_activations \\
    --pca_files        .../fineweb/fineweb_pca_components_Qwen3-8B_20000.pt \\
    --output_dir       results/cosine_probe_deconf

# With multiple PCA files (deconfounded separately per file):
python eval_cosine_similarity_probe_prm800k_with_deconfounding.py \\
    --mode both \\
    --datasets_vectors .../processbench/... .../prm800k/... \\
    --datasets_raw     .../processbench/raw_activations .../prm800k/raw_activations \\
    --pca_files \\
        .../fineweb/fineweb_pca_components_Qwen3-8B_20000.pt \\
        .../deepmind_math/deepmind_math_pca_components_Qwen3-8B_20000.pt \\
    --output_dir       results/cosine_probe_deconf
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
        description=(
            "Cosine-similarity probe evaluation for reasoning directions, "
            "with optional PCA-based deconfounding."
        )
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
    # ---- PCA deconfounding ----
    p.add_argument(
        "--pca_files", type=str, nargs="*", default=[
            "/home/ines/Reasoning-activations/baseline_vectors/Qwen3-8B/deepmind_math_activations_20000.pt",
            "/home/ines/Reasoning-activations/baseline_vectors/Qwen3-8B/fineweb_pca_components.pt"

        ],
        help=(
            "Zero or more paths to PCA component .pt files produced by "
            "run_fw_pass_with_step_averaging_storage_v2.py --type baseline. "
            "Each file is used to independently deconfound the activations. "
            "File format: {'layers': {<int layer>: Tensor[n_comp, d_model]}, 'metadata': {...}}. "
            "If not supplied, only the original (non-deconfounded) analysis is run."
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
# PCA deconfounding helpers
# ============================================================
def load_pca_components(pca_path: str) -> dict:
    """
    Load a PCA component file produced by run_fw_pass_with_step_averaging_storage_v2.py
    --type baseline.

    Returns a dict mapping layer index (int) -> np.ndarray[n_components, d_model] float32.
    Supports both integer and string-keyed 'layers' dicts.
    """
    pca_data = torch.load(pca_path, weights_only=False, map_location="cpu")
    raw_layers = pca_data["layers"]
    components: dict[int, np.ndarray] = {}
    for key, val in raw_layers.items():
        # key can be int (e.g. 18) or string (e.g. "blocks.18.hook_out")
        if isinstance(key, int):
            layer_idx = key
        else:
            try:
                layer_idx = int(str(key).split(".")[1])
            except (IndexError, ValueError):
                # Try parsing the whole string as an int
                try:
                    layer_idx = int(key)
                except ValueError:
                    print(f"    [PCA] WARNING: Cannot parse layer key '{key}', skipping.")
                    continue
        components[layer_idx] = val.to(torch.float32).numpy()
    return components


def get_pca_components_for_layer(
    pca_components: dict,
    layer: int,
    hook_name: str | None = None,
) -> np.ndarray | None:
    """
    Look up PCA components for a given layer index.
    Falls back to hook_name string lookup if integer key is missing.

    Returns np.ndarray[n_components, d_model] or None if not found.
    """
    if layer in pca_components:
        return pca_components[layer]
    # Fallback: search by integer parsed from hook_name string
    if hook_name is not None:
        try:
            idx = int(hook_name.split(".")[1])
            if idx in pca_components:
                return pca_components[idx]
        except (IndexError, ValueError):
            pass
    return None


def project_out_components(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Project out a set of (unit-norm, orthogonal) PCA components from a matrix of
    row vectors.

    Implements the same logic as deconfounding_step_reasoning_vectors.ipynb:
        for each component c_i:
            x ← x - (x @ c_i) * c_i

    Parameters
    ----------
    X          : np.ndarray, shape (N, d_model) or (d_model,)
    components : np.ndarray, shape (n_components, d_model), rows are unit vectors

    Returns
    -------
    np.ndarray of same shape as X, float32
    """
    x = X.astype(np.float32).copy()
    original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
    for comp in components:          # comp: [d_model]
        proj = x @ comp              # [N]  dot product of each row with component
        x -= proj[:, None] * comp[None, :]   # subtract projection
    return x.reshape(original_shape)


def deconfound_dataset_data(
    dataset_data: dict,
    pca_components: dict,
    layer: int,
    hook_name: str,
) -> dict | None:
    """
    Apply PCA deconfounding to step_X and sample_X for a single layer.

    Returns a new dict with the same structure as dataset_data but with
    deconfounded step_X and sample_X, or None if no PCA components are
    available for this layer.
    """
    comps = get_pca_components_for_layer(pca_components, layer, hook_name)
    if comps is None:
        return None

    step_X_deconf   = project_out_components(dataset_data["step_X"],   comps)
    sample_X_deconf = project_out_components(dataset_data["sample_X"], comps)

    return {
        "step_X":       step_X_deconf,
        "step_y":       dataset_data["step_y"],
        "step_groups":  dataset_data["step_groups"],
        "sample_X":     sample_X_deconf,
        "sample_y":     dataset_data["sample_y"],
        "sample_groups": dataset_data.get(
            "sample_groups",
            np.arange(len(dataset_data["sample_y"]), dtype=np.int32)
        ),
    }


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
    suffix: str = "",
):
    """
    Compute and save the full-dataset reasoning direction vectors (step and sample
    granularity) as .pt tensors, in both raw and unit-normalised forms.

    suffix : appended to filenames for deconfounded variants
             e.g. "_deconf_fineweb_pca_components"

    Output layout:
        <output_dir>/<dataset_tag>/direction_vectors/layer_<layer>/
            reasoning_direction_step_raw[suffix].pt
            reasoning_direction_step_unit[suffix].pt
            reasoning_direction_sample_raw[suffix].pt
            reasoning_direction_sample_unit[suffix].pt
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
            (raw,  f"reasoning_direction_{granularity}_raw{suffix}.pt"),
            (unit, f"reasoning_direction_{granularity}_unit{suffix}.pt"),
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
def print_layer_results(layer: int, layer_results: dict, do_cv: bool, do_holdout: bool,
                        label_prefix: str = ""):
    print(f"\n{'='*70}")
    title = f"  LAYER {layer}"
    if label_prefix:
        title += f"  [{label_prefix}]"
    print(title)
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


def _pca_tag_from_path(path: str) -> str:
    """Derive a short human-readable tag from a PCA file path (stem without extension)."""
    return Path(path).stem


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
    title_prefix: str = "",
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
            full_title = f"  SUMMARY TABLE -- {title_prefix}{title}  [{scorer_name}]"
            print(full_title)
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
# Core per-layer evaluation (original or deconfounded)
# ============================================================
def run_layer_evaluation(
    *,
    layer: int,
    dataset_tags: list[str],
    dataset_data: dict,
    vec_paths: list[str],
    raw_paths: list[str],
    ho_pairs: list[tuple],
    do_cv: bool,
    do_holdout: bool,
    n_folds: int,
    seed: int,
    output_dir: Path,
    save_directions: bool,
    cv_results: dict,
    holdout_results: dict,
    direction_suffix: str = "",
    label_prefix: str = "",
):
    """
    Run CV and holdout evaluation for a single layer, updating cv_results and
    holdout_results in place.

    direction_suffix : appended to saved direction-vector filenames
    label_prefix     : prepended to printed layer result headers
    """
    # Save direction vectors (original or deconfounded variant)
    if save_directions:
        for tag in dataset_tags:
            save_direction_vectors(
                output_dir=output_dir,
                dataset_tag=tag,
                layer=layer,
                step_X=dataset_data[tag]["step_X"],
                step_y=dataset_data[tag]["step_y"],
                sample_X=dataset_data[tag]["sample_X"],
                sample_y=dataset_data[tag]["sample_y"],
                suffix=direction_suffix,
            )

    # CV
    if do_cv:
        print(f"\n  {'='*60}")
        print(f"  CV evaluation{f'  [{label_prefix}]' if label_prefix else ''}")
        print(f"  {'='*60}")
        for tag in dataset_tags:
            print(f"\n  CV on: {tag}")
            cv_layer = evaluate_layer_cv(
                layer=layer,
                train_data=dataset_data[tag],
                n_folds=n_folds,
                seed=seed,
            )
            cv_results[tag][str(layer)] = cv_layer
            print_layer_results(layer, cv_layer, do_cv=True, do_holdout=False,
                                label_prefix=label_prefix)

    # Holdout
    if do_holdout:
        print(f"\n  {'='*60}")
        print(f"  Hold-out evaluation{f'  [{label_prefix}]' if label_prefix else ''} (all ordered pairs)")
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
            print_layer_results(layer, ho_layer, do_cv=False, do_holdout=True,
                                label_prefix=label_prefix)


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

    # Load PCA files
    pca_files = args.pca_files or []
    pca_bundles: list[tuple[str, dict]] = []   # (tag, {layer_idx: np.ndarray})
    for pca_path in pca_files:
        pca_tag = _pca_tag_from_path(pca_path)
        print(f"\nLoading PCA components: {pca_path}  (tag='{pca_tag}')")
        comps = load_pca_components(pca_path)
        print(f"  Loaded PCA for layers: {sorted(comps.keys())}  "
              f"  n_components per layer: "
              f"{[comps[k].shape[0] for k in sorted(comps.keys())]}")
        pca_bundles.append((pca_tag, comps))

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
    if pca_bundles:
        print(f"  PCA deconf files : {len(pca_bundles)}")
        for tag, _ in pca_bundles:
            print(f"    {tag}")
    else:
        print(f"  PCA deconf files : none (original analysis only)")
    print(f"{'#'*70}\n")

    # Pre-load all vector files once
    print("Pre-loading all vector files ...")
    vec_cache: dict[str, dict] = {}
    for vp in vec_paths:
        if vp not in vec_cache:
            print(f"  Loading {vp}")
            vec_cache[vp] = torch.load(vp, weights_only=False, map_location="cpu")

    # Results containers — original
    cv_results: dict[str, dict] = {tag: {} for tag in dataset_tags}
    ho_pairs = list(permutations(range(n_datasets), 2)) if do_holdout else []
    holdout_results: dict[tuple, dict] = {
        (dataset_tags[i], dataset_tags[j]): {} for i, j in ho_pairs
    }

    # Results containers — one per PCA file
    deconf_cv_results:      list[dict] = [{tag: {} for tag in dataset_tags} for _ in pca_bundles]
    deconf_holdout_results: list[dict] = [
        {(dataset_tags[i], dataset_tags[j]): {} for i, j in ho_pairs}
        for _ in pca_bundles
    ]

    # ------------------------------------------------------------------
    # Per-layer loop
    # ------------------------------------------------------------------
    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'#'*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'#'*70}")

        # ----- Aggregate raw activations for all datasets -----
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

        # ----- Original (non-deconfounded) evaluation -----
        print(f"\n{'='*60}")
        print(f"  ORIGINAL (non-deconfounded) evaluation — Layer {layer}")
        print(f"{'='*60}")

        run_layer_evaluation(
            layer=layer,
            dataset_tags=dataset_tags,
            dataset_data=dataset_data,
            vec_paths=vec_paths,
            raw_paths=raw_paths,
            ho_pairs=ho_pairs,
            do_cv=do_cv,
            do_holdout=do_holdout,
            n_folds=args.n_folds,
            seed=args.seed,
            output_dir=output_dir,
            save_directions=args.save_directions,
            cv_results=cv_results,
            holdout_results=holdout_results,
            direction_suffix="",
            label_prefix="original",
        )

        # ----- Deconfounded evaluation (once per PCA file) -----
        for pca_idx, (pca_tag, pca_components) in enumerate(pca_bundles):
            print(f"\n{'='*60}")
            print(f"  DECONFOUNDED [{pca_tag}] — Layer {layer}")
            print(f"{'='*60}")

            # Apply deconfounding to each dataset's activations
            deconf_data: dict[str, dict] = {}
            skipped_tags: list[str] = []
            for tag in dataset_tags:
                dc = deconfound_dataset_data(
                    dataset_data=dataset_data[tag],
                    pca_components=pca_components,
                    layer=layer,
                    hook_name=hook_name,
                )
                if dc is None:
                    print(f"    WARNING: No PCA components found for layer {layer} "
                          f"in '{pca_tag}'.  Skipping deconfounded analysis for {tag}.")
                    skipped_tags.append(tag)
                else:
                    n_comp = pca_components.get(layer, np.empty((0,))).shape[0]
                    print(f"    [{tag}] Deconfounded with {n_comp} components "
                          f"from '{pca_tag}'.")
                    deconf_data[tag] = dc

            if not deconf_data:
                print(f"    No datasets could be deconfounded for layer {layer} "
                      f"with '{pca_tag}'. Skipping.")
                continue

            # Restrict evaluation to datasets for which deconfounding succeeded
            available_tags = [t for t in dataset_tags if t in deconf_data]
            available_indices = [dataset_tags.index(t) for t in available_tags]
            available_ho_pairs = [
                (i, j) for (i, j) in ho_pairs
                if dataset_tags[i] in deconf_data and dataset_tags[j] in deconf_data
            ]

            run_layer_evaluation(
                layer=layer,
                dataset_tags=available_tags,
                dataset_data=deconf_data,
                vec_paths=[vec_paths[k] for k in available_indices],
                raw_paths=[raw_paths[k] for k in available_indices],
                ho_pairs=available_ho_pairs,
                do_cv=do_cv,
                do_holdout=do_holdout,
                n_folds=args.n_folds,
                seed=args.seed,
                output_dir=output_dir,
                save_directions=args.save_directions,
                cv_results=deconf_cv_results[pca_idx],
                holdout_results=deconf_holdout_results[pca_idx],
                direction_suffix=f"_deconf_{pca_tag}",
                label_prefix=f"deconf:{pca_tag}",
            )

            del deconf_data

        del dataset_data
        gc.collect()

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------

    # Original results
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

    # Deconfounded results (one file per PCA source)
    for pca_idx, (pca_tag, _) in enumerate(pca_bundles):
        dc_cv  = deconf_cv_results[pca_idx]
        dc_ho  = deconf_holdout_results[pca_idx]

        if do_cv:
            for tag in dataset_tags:
                if not dc_cv.get(tag):
                    continue
                tag_dir = output_dir / tag
                tag_dir.mkdir(parents=True, exist_ok=True)
                out_path = tag_dir / f"cosine_probe_cv_results_deconf_{pca_tag}.json"
                with open(out_path, "w") as f:
                    json.dump(to_serialisable(dc_cv[tag]), f, indent=2)
                print(f"\nCV results [deconf:{pca_tag}] saved -> {out_path}")

        if do_holdout:
            for (train_tag, eval_tag), ho_res in dc_ho.items():
                if not ho_res:
                    continue
                ho_dir = output_dir / train_tag / f"eval_on_{eval_tag}"
                ho_dir.mkdir(parents=True, exist_ok=True)
                out_path = ho_dir / f"cosine_probe_holdout_results_deconf_{pca_tag}.json"
                with open(out_path, "w") as f:
                    json.dump(to_serialisable(ho_res), f, indent=2)
                print(f"Holdout results [deconf:{pca_tag}] saved -> {out_path}")

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    print_summary_tables(
        cv_results=cv_results,
        holdout_results=holdout_results,
        dataset_tags=dataset_tags,
        layers=args.target_layers,
        do_cv=do_cv,
        do_holdout=do_holdout,
        title_prefix="[ORIGINAL] ",
    )

    for pca_idx, (pca_tag, _) in enumerate(pca_bundles):
        dc_cv = deconf_cv_results[pca_idx]
        dc_ho = deconf_holdout_results[pca_idx]
        available_tags = [t for t in dataset_tags if dc_cv.get(t)]
        print_summary_tables(
            cv_results=dc_cv,
            holdout_results=dc_ho,
            dataset_tags=available_tags,
            layers=args.target_layers,
            do_cv=do_cv,
            do_holdout=do_holdout,
            title_prefix=f"[DECONF:{pca_tag}] ",
        )


if __name__ == "__main__":
    main()