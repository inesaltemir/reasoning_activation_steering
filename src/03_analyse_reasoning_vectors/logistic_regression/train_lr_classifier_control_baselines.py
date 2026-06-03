"""
Control Baselines for LR Classifier on Reasoning Activations
==============================================================

This script runs alongside train_lr_classifier_streaming_extended.py to verify
that the logistic-regression classifier is capturing *genuine reasoning quality*
rather than superficial confounds (position, length, topic, high-dim chance).

Six control baselines, applied at each granularity level:

  1. PERMUTATION TEST (shuffled labels)
     Shuffle y labels N times, retrain the same pipeline each time.
     The real classifier's AUROC should be well above the permuted distribution.
     → If not, the original result is likely spurious.

  2. STEP-POSITION-ONLY BASELINE
     Train LR using only step_idx (or sample token-count rank) as the single feature.
     → If AUROC is high, position alone explains the classification.

  3. TOKEN-COUNT-ONLY BASELINE
     Train LR using only the number of tokens aggregated per step/sample.
     → If AUROC is high, length is a dominant confound.

  4. POSITION-RESIDUALIZED ACTIVATIONS
     Regress out step_idx from each activation dimension, then train LR on residuals.
     → If AUROC drops substantially vs. the real classifier, the signal was positional.

  5. WITHIN-PROBLEM CENTERED (topic control, step-level only)
     Subtract per-problem mean activation so that problem identity is removed.
     Only keep problems that have BOTH correct and incorrect steps.
     → If AUROC drops, the classifier was partly learning problem identity.

  6. RANDOM PROJECTION BASELINE
     Project activations onto a random unit vector, use the scalar as a 1D classifier.
     → Should give ~0.5 AUROC. Confirms the real direction is special.

"""

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression
import warnings
import gc

warnings.filterwarnings("ignore")


# ==========================================
# Args
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Control baselines for LR reasoning classifier."
    )
    p.add_argument("--raw_dir", type=str,
                   default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations")
    p.add_argument("--vectors_file", type=str,
                   default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/"
                           "reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt")
    p.add_argument("--target_layers", type=int, nargs="+",
                   default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--granularities", type=str, nargs="+",
                   default=["step"],
                   choices=["token", "step", "sample"])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--sgd_epochs", type=int, default=5)
    p.add_argument("--test_shards", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_controls")
    # Control-specific
    p.add_argument("--n_permutations", type=int, default=20,
                   help="Number of label permutations for the permutation test")
    p.add_argument("--n_random_projections", type=int, default=10,
                   help="Number of random directions for baseline 6")
    return p.parse_args()


# ==========================================
# Shard loader (reused from main script)
# ==========================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    safe = hook_name.replace(".", "_")
    acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt",
                      weights_only=False).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt",
                      weights_only=False)
    return acts, meta


# ==========================================
# Shared: quick CV logistic regression
# ==========================================
def _fit_cv_lr(X, y, args, groups=None, C_override=None):
    """Stratified K-fold CV on StandardScaler → LR.  Returns metrics dict."""
    if len(np.unique(y)) < 2:
        return {"error": "single_class", "cv_roc_auc_mean": 0.5}

    C = C_override if C_override is not None else args.C
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=args.seed)),
    ])

    scoring = ["accuracy", "roc_auc", "f1"]

    if groups is not None:
        n_groups = len(np.unique(groups))
        n_folds = min(args.n_folds, min(np.bincount(y)), n_groups)
        n_folds = max(n_folds, 2)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                    return_train_score=True, groups=groups)
    else:
        n_folds = min(args.n_folds, min(np.bincount(y)))
        n_folds = max(n_folds, 2)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                    return_train_score=True)
        
    metrics = {}
    for s in scoring:
        metrics[f"cv_{s}_mean"] = float(np.mean(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_std"]  = float(np.std(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_train_mean"] = float(np.mean(cv_results[f"train_{s}"]))
        metrics[f"cv_{s}_train_std"]  = float(np.std(cv_results[f"train_{s}"]))

    metrics["n_samples"]  = int(len(y))
    metrics["n_positive"] = int(y.sum())
    metrics["n_negative"] = int(len(y) - y.sum())
    metrics["n_folds"]    = n_folds
    return metrics


# ==========================================
# Step-level data loader (streams shards)
# ==========================================
def load_step_level_data(raw_dir, hook_name, num_shards, d_model):
    """Stream all shards and aggregate to step-level.
    Returns X, y, keys, token_counts, step_positions, sample_ids."""
    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)
    step_positions = {}  # (sample_idx, step_idx) → step_idx

    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)
        for i, m in enumerate(meta):
            if m["step_idx"] < 0 or m["is_correct"] is None:
                continue
            key = (m["sample_idx"], m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if m["is_correct"] is False:
                step_labels[key] = False
            step_positions[key] = m["step_idx"]
        del acts, acts_np, meta; gc.collect()

    keys = sorted(step_sums.keys())
    X = np.stack([step_sums[k] / step_counts[k] for k in keys]).astype(np.float32)
    y = np.array([1 if step_labels[k] else 0 for k in keys], dtype=np.int32)
    groups = np.array([k[0] for k in keys])  # sample_idx per step
    counts = np.array([step_counts[k] for k in keys], dtype=np.int32)
    positions = np.array([step_positions[k] for k in keys], dtype=np.int32)
    sample_ids = np.array([k[0] for k in keys], dtype=np.int32)

    del step_sums, step_counts, step_labels, step_positions; gc.collect()
    return X, y, keys, counts, positions, sample_ids, groups


# ==========================================
# Sample-level data loader
# ==========================================
def load_sample_level_data(raw_dir, hook_name, num_shards, d_model,
                            per_sample_is_fully_correct):
    """Stream all shards and aggregate to sample-level."""
    sample_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    sample_counts = defaultdict(int)

    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)
        for i, m in enumerate(meta):
            s = m["sample_idx"]
            sample_sums[s]   += acts_np[i]
            sample_counts[s] += 1
        del acts, acts_np, meta; gc.collect()

    sample_ids = sorted(sample_sums.keys())
    X = np.stack([sample_sums[s] / sample_counts[s] for s in sample_ids]).astype(np.float32)
    y = np.array([
        int(per_sample_is_fully_correct[s].item())
        for s in sample_ids
        if s < len(per_sample_is_fully_correct)
    ], dtype=np.int32)
    X = X[:len(y)]
    counts = np.array([sample_counts[s] for s in sample_ids[:len(y)]], dtype=np.int32)

    del sample_sums, sample_counts; gc.collect()
    return X, y, sample_ids[:len(y)], counts


# ==========================================
# Token-level data loader (loads all into RAM
# for control tests — not streaming SGD)
# ==========================================
def load_token_level_data(raw_dir, hook_name, num_shards, max_tokens=500_000):
    """Load token-level data.  For tractability in controls, cap at max_tokens.
    Returns X, y, step_positions, sample_ids, token_counts_per_step."""
    all_X, all_y, all_pos, all_sid = [], [], [], []

    for s_id in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, s_id)
        for i, m in enumerate(meta):
            if m["is_correct"] is None:
                continue
            all_X.append(acts[i].numpy())
            all_y.append(1 if m["is_correct"] else 0)
            all_pos.append(m.get("step_idx", -1))
            all_sid.append(m["sample_idx"])
            if len(all_y) >= max_tokens:
                break
        del acts, meta; gc.collect()
        if len(all_y) >= max_tokens:
            break

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)
    positions = np.array(all_pos, dtype=np.int32)
    sample_ids = np.array(all_sid, dtype=np.int32)

    del all_X, all_y, all_pos, all_sid; gc.collect()
    return X, y, positions, sample_ids


# ==================================================================
# CONTROL 1: Permutation test
# ==================================================================
# too costly, don't do it
def control_permutation(X, y, args, n_perms):
    """Shuffle labels n_perms times, fit LR each time.
    Returns distribution of null AUROC values."""
    rng = np.random.RandomState(args.seed)
    null_aurocs = []
    null_accs = []

    for i in range(n_perms):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        m = _fit_cv_lr(X, y_perm, args)
        null_aurocs.append(m["cv_roc_auc_mean"])
        null_accs.append(m["cv_accuracy_mean"])
        print(f"      Permutation {i+1}/{n_perms}: AUROC={m['cv_roc_auc_mean']:.4f}  "
              f"Acc={m['cv_accuracy_mean']:.4f}")

    return {
        "null_aurocs": null_aurocs,
        "null_auroc_mean": float(np.mean(null_aurocs)),
        "null_auroc_std":  float(np.std(null_aurocs)),
        "null_auroc_max":  float(np.max(null_aurocs)),
        "null_accs": null_accs,
        "null_acc_mean": float(np.mean(null_accs)),
        "null_acc_std":  float(np.std(null_accs)),
        "n_permutations": n_perms,
    }


# ==================================================================
# CONTROL 2: Step-position-only baseline
# ==================================================================
def control_position_only(positions, y, args):
    """Train LR using only step position as the feature."""
    X_pos = positions.reshape(-1, 1).astype(np.float32)
    metrics = _fit_cv_lr(X_pos, y, args, C_override=10.0)
    metrics["description"] = "LR trained on step_idx only (1D feature)"
    return metrics

# For control 2 & 3, do i want stratified groups? nop
# should it be balanced?

# ==================================================================
# CONTROL 3: Token-count-only baseline
# ==================================================================
def control_token_count_only(token_counts, y, args):
    """Train LR using only the token count as the feature."""
    X_len = token_counts.reshape(-1, 1).astype(np.float32)
    metrics = _fit_cv_lr(X_len, y, args, C_override=10.0)
    metrics["description"] = "LR trained on token_count only (1D feature)"

    # Also report basic stats
    metrics["token_count_correct_mean"] = float(np.mean(token_counts[y == 1]))
    metrics["token_count_correct_std"]  = float(np.std(token_counts[y == 1]))
    metrics["token_count_incorrect_mean"] = float(np.mean(token_counts[y == 0]))
    metrics["token_count_incorrect_std"]  = float(np.std(token_counts[y == 0]))
    return metrics


# ==================================================================
# CONTROL 4: Position-residualized activations
# ==================================================================
def control_position_residualized(X, y, positions, args):
    """Regress out step position from each activation dimension,
    then train LR on the residuals."""
    X_pos = positions.reshape(-1, 1).astype(np.float64)

    # Fit a linear model: activation_dim_j = a_j * step_idx + b_j
    # Then take residuals
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_pos, X.astype(np.float64))
    X_predicted = reg.predict(X_pos)
    X_residual = (X.astype(np.float64) - X_predicted).astype(np.float32)

    # reconstruct the step idx from linear reg
    # prob some corr between step idx 
    # how good is it

    metrics = _fit_cv_lr(X_residual, y, args)
    metrics["description"] = "LR trained on activations after regressing out step_idx"

    # Also report variance explained by position
    ss_total = np.sum((X.astype(np.float64) - X.mean(axis=0)) ** 2)
    ss_resid = np.sum(X_residual.astype(np.float64) ** 2)
    r2_position = 1.0 - ss_resid / (ss_total + 1e-12)
    metrics["r2_position_explains"] = float(r2_position)

    del X_residual, X_predicted; gc.collect()
    return metrics


# ==================================================================
# CONTROL 5: Within-problem centered (topic control)
# ==================================================================
def control_within_problem(X, y, sample_ids, args):
    """Subtract per-problem mean activation and keep only problems
    that have BOTH correct and incorrect steps."""
    unique_samples = np.unique(sample_ids)

    # Find problems with both classes
    mixed_problems = []
    for s in unique_samples:
        mask = sample_ids == s
        labels_in_problem = y[mask]
        if len(np.unique(labels_in_problem)) == 2:
            mixed_problems.append(s)

    if len(mixed_problems) < 5:
        return {
            "error": "too_few_mixed_problems",
            "n_mixed_problems": len(mixed_problems),
            "description": "Not enough problems with both correct and incorrect steps"
        }

    mixed_set = set(mixed_problems)
    keep_mask = np.array([s in mixed_set for s in sample_ids])

    X_mixed = X[keep_mask].copy()
    y_mixed = y[keep_mask].copy()
    sid_mixed = sample_ids[keep_mask]

    # Subtract per-problem mean
    for s in mixed_problems:
        mask = sid_mixed == s
        problem_mean = X_mixed[mask].mean(axis=0)
        X_mixed[mask] -= problem_mean

    metrics = _fit_cv_lr(X_mixed, y_mixed, args)
    metrics["description"] = (
        "LR on within-problem centered activations "
        "(per-problem mean subtracted, only mixed-label problems)"
    )
    metrics["n_mixed_problems"] = len(mixed_problems)
    metrics["n_total_problems"] = len(unique_samples)
    metrics["n_steps_kept"] = int(keep_mask.sum())
    metrics["n_steps_total"] = int(len(y))

    del X_mixed, y_mixed; gc.collect()
    return metrics


# ==================================================================
# CONTROL 6: Random projection baseline
# ==================================================================
def control_random_projection(X, y, args, n_projections):
    """Project activations onto random unit vectors, use as 1D classifier."""
    rng = np.random.RandomState(args.seed + 999)
    d = X.shape[1]
    aurocs = []

    for i in range(n_projections):
        direction = rng.randn(d).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-12

        scores = X @ direction
        # Simple AUROC on the scalar projection (no LR needed)
        if len(np.unique(y)) < 2:
            aurocs.append(0.5)
            continue
        auroc = roc_auc_score(y, scores)
        # AUROC can be < 0.5 if direction is flipped; take max(auroc, 1-auroc)
        auroc = max(auroc, 1.0 - auroc)
        aurocs.append(auroc)

    return {
        "random_aurocs": aurocs,
        "random_auroc_mean": float(np.mean(aurocs)),
        "random_auroc_std":  float(np.std(aurocs)),
        "random_auroc_max":  float(np.max(aurocs)),
        "n_projections": n_projections,
        "description": "AUROC from projecting activations onto random unit vectors",
    }


# ==================================================================
# CONTROL 7 (bonus): Position + length combined baseline
# ==================================================================
def control_position_and_length(positions, token_counts, y, args):
    """Train LR using both step position and token count (2D)."""
    X_cov = np.stack([positions.astype(np.float32),
                      token_counts.astype(np.float32)], axis=1)
    metrics = _fit_cv_lr(X_cov, y, args, C_override=10.0)
    metrics["description"] = "LR trained on [step_idx, token_count] (2D features)"
    return metrics


# ==================================================================
# Run all controls for one (layer, granularity)
# ==================================================================
def run_controls_step_level(raw_dir, hook_name, num_shards, d_model, args):
    """All controls for step-level granularity."""
    print(f"    Loading step-level data...")
    X, y, keys, counts, positions, sample_ids, groups = load_step_level_data(
        raw_dir, hook_name, num_shards, d_model)
    print(f"    Steps: {len(y)} | Pos: {y.sum()} | Neg: {len(y)-y.sum()}")

    if len(np.unique(y)) < 2:
        return {"error": "single_class"}

    # Real classifier baseline (for comparison)
    print(f"    [Real] Fitting LR on full activations...")
    real_metrics = _fit_cv_lr(X, y, args, groups)

    results = {"real_classifier": real_metrics}

    # Control 1: Permutation test
    #print(f"    [Control 1] Permutation test ({args.n_permutations} permutations)...")
    #results["ctrl1_permutation"] = control_permutation(X, y, args, args.n_permutations)
    #real_auroc = real_metrics["cv_roc_auc_mean"]
    #null_aurocs = results["ctrl1_permutation"]["null_aurocs"]
    #p_value = float(np.mean([na >= real_auroc for na in null_aurocs]))
    #results["ctrl1_permutation"]["empirical_p_value"] = p_value
    #print(f"      Real AUROC: {real_auroc:.4f} | Null mean: "
    #      f"{results['ctrl1_permutation']['null_auroc_mean']:.4f} | p={p_value:.3f}")

    # Control 2: Position-only
    print(f"    [Control 2] Position-only baseline...")
    results["ctrl2_position_only"] = control_position_only(positions, y, args)
    print(f"      Position-only AUROC: {results['ctrl2_position_only']['cv_roc_auc_mean']:.4f}")

    # Control 3: Token-count-only
    print(f"    [Control 3] Token-count-only baseline...")
    results["ctrl3_token_count_only"] = control_token_count_only(counts, y, args)
    print(f"      Token-count-only AUROC: {results['ctrl3_token_count_only']['cv_roc_auc_mean']:.4f}")

    # Control 4: Position-residualized
    print(f"    [Control 4] Position-residualized activations...")
    results["ctrl4_position_residualized"] = control_position_residualized(X, y, positions, args)
    print(f"      Residualized AUROC: {results['ctrl4_position_residualized']['cv_roc_auc_mean']:.4f}")
    print(f"      R² explained by position: "
          f"{results['ctrl4_position_residualized']['r2_position_explains']:.4f}")

    # Control 5: Within-problem centered
    print(f"    [Control 5] Within-problem centered (topic control)...")
    results["ctrl5_within_problem"] = control_within_problem(X, y, sample_ids, args)
    if "error" not in results["ctrl5_within_problem"]:
        print(f"      Within-problem AUROC: {results['ctrl5_within_problem']['cv_roc_auc_mean']:.4f} "
              f"({results['ctrl5_within_problem']['n_mixed_problems']} mixed problems)")
    else:
        print(f"      Skipped: {results['ctrl5_within_problem']['error']}")

    # Control 6: Random projection
    print(f"    [Control 6] Random projection baseline ({args.n_random_projections} directions)...")
    results["ctrl6_random_projection"] = control_random_projection(X, y, args, args.n_random_projections)
    print(f"      Random AUROC: {results['ctrl6_random_projection']['random_auroc_mean']:.4f} "
          f"± {results['ctrl6_random_projection']['random_auroc_std']:.4f}")

    # Control 7: Position + length combined
    print(f"    [Control 7] Position + token-count combined (2D)...")
    results["ctrl7_position_and_length"] = control_position_and_length(positions, counts, y, args)
    print(f"      Position+Length AUROC: {results['ctrl7_position_and_length']['cv_roc_auc_mean']:.4f}")

    del X, y; gc.collect()
    return results


def run_controls_sample_level(raw_dir, hook_name, num_shards, d_model,
                               per_sample_mask, args):
    """All controls for sample-level granularity.
    For the controls who do not depend on activations (step idx and length)
    enough to run for one layer only
    """
    print(f"    Loading sample-level data...")
    X, y, sample_ids, counts = load_sample_level_data(
        raw_dir, hook_name, num_shards, d_model, per_sample_mask)
    print(f"    Samples: {len(y)} | Pos: {y.sum()} | Neg: {len(y)-y.sum()}")

    if len(np.unique(y)) < 2:
        return {"error": "single_class"}

    # Real classifier
    print(f"    [Real] Fitting LR on full activations...")
    real_metrics = _fit_cv_lr(X, y, args)
    results = {"real_classifier": real_metrics}

    # Control 1: Permutation
    print(f"    [Control 1] Permutation test...")
    results["ctrl1_permutation"] = control_permutation(X, y, args, args.n_permutations)
    real_auroc = real_metrics["cv_roc_auc_mean"]
    null_aurocs = results["ctrl1_permutation"]["null_aurocs"]
    p_value = float(np.mean([na >= real_auroc for na in null_aurocs]))
    results["ctrl1_permutation"]["empirical_p_value"] = p_value
    print(f"      Real AUROC: {real_auroc:.4f} | Null: "
          f"{results['ctrl1_permutation']['null_auroc_mean']:.4f} | p={p_value:.3f}")

    # Control 3: Token-count-only (no position for sample-level)
    print(f"    [Control 3] Token-count-only baseline...")
    results["ctrl3_token_count_only"] = control_token_count_only(counts, y, args)
    print(f"      Token-count-only AUROC: {results['ctrl3_token_count_only']['cv_roc_auc_mean']:.4f}")

    # Control 6: Random projection
    print(f"    [Control 6] Random projection baseline...")
    results["ctrl6_random_projection"] = control_random_projection(X, y, args, args.n_random_projections)
    print(f"      Random AUROC: {results['ctrl6_random_projection']['random_auroc_mean']:.4f} "
          f"± {results['ctrl6_random_projection']['random_auroc_std']:.4f}")

    del X, y; gc.collect()
    return results


def run_controls_token_level(raw_dir, hook_name, num_shards, args,
                              max_tokens=300_000):
    """Controls for token-level.  Loads a capped subset into RAM."""
    print(f"    Loading token-level data (max {max_tokens:,} tokens)...")
    X, y, positions, sample_ids = load_token_level_data(
        raw_dir, hook_name, num_shards, max_tokens=max_tokens)
    print(f"    Tokens: {len(y)} | Pos: {y.sum()} | Neg: {len(y)-y.sum()}")

    if len(np.unique(y)) < 2:
        return {"error": "single_class"}

    # Real classifier (full LR, not SGD, on this subset for comparability)
    print(f"    [Real] Fitting LR on token activations (subset)...")
    real_metrics = _fit_cv_lr(X, y, args)
    results = {"real_classifier_subset": real_metrics}

    # Control 1: Permutation (fewer perms for speed)
    n_perms = min(args.n_permutations, 10)
    print(f"    [Control 1] Permutation test ({n_perms} permutations)...")
    results["ctrl1_permutation"] = control_permutation(X, y, args, n_perms)
    real_auroc = real_metrics["cv_roc_auc_mean"]
    null_aurocs = results["ctrl1_permutation"]["null_aurocs"]
    p_value = float(np.mean([na >= real_auroc for na in null_aurocs]))
    results["ctrl1_permutation"]["empirical_p_value"] = p_value

    # Control 2: Position-only
    print(f"    [Control 2] Position-only baseline...")
    results["ctrl2_position_only"] = control_position_only(positions, y, args)
    print(f"      Position-only AUROC: {results['ctrl2_position_only']['cv_roc_auc_mean']:.4f}")

    # Control 4: Position-residualized
    print(f"    [Control 4] Position-residualized activations...")
    results["ctrl4_position_residualized"] = control_position_residualized(X, y, positions, args)
    print(f"      Residualized AUROC: {results['ctrl4_position_residualized']['cv_roc_auc_mean']:.4f}")

    # Control 5: Within-problem centered
    print(f"    [Control 5] Within-problem centered (topic control)...")
    results["ctrl5_within_problem"] = control_within_problem(X, y, sample_ids, args)

    # Control 6: Random projection
    print(f"    [Control 6] Random projection baseline...")
    results["ctrl6_random_projection"] = control_random_projection(X, y, args, args.n_random_projections)
    print(f"      Random AUROC: {results['ctrl6_random_projection']['random_auroc_mean']:.4f}")

    del X, y; gc.collect()
    return results


# ==================================================================
# Summary report
# ==================================================================
def print_summary(all_results):
    """Print a comparison table across all layers and granularities."""
    print(f"\n{'='*90}")
    print(f"  CONTROL BASELINES SUMMARY")
    print(f"{'='*90}")

    for layer_key in sorted(all_results.keys(), key=lambda x: int(x)):
        print(f"\n  Layer {layer_key}")
        print(f"  {'-'*80}")

        for gran, controls in all_results[layer_key].items():
            if isinstance(controls, dict) and "error" in controls:
                print(f"    {gran:>8}: ERROR — {controls['error']}")
                continue

            real_auroc = controls.get("real_classifier", controls.get("real_classifier_subset", {}))
            real_auc = real_auroc.get("cv_roc_auc_mean", "N/A")

            print(f"    {gran:>8} | Real AUROC: {real_auc:.4f}", end="")

            # Permutation
            #perm = controls.get("ctrl1_permutation", {})
            #if perm:
            #    print(f" | Null: {perm.get('null_auroc_mean', 0):.4f} "
            #          f"(p={perm.get('empirical_p_value', 1):.3f})", end="")

            # Position
            pos = controls.get("ctrl2_position_only", {})
            if pos and "error" not in pos:
                print(f" | Pos-only: {pos['cv_roc_auc_mean']:.4f}", end="")

            # Token count
            tc = controls.get("ctrl3_token_count_only", {})
            if tc and "error" not in tc:
                print(f" | Len-only: {tc['cv_roc_auc_mean']:.4f}", end="")

            # Residualized
            res = controls.get("ctrl4_position_residualized", {})
            if res and "error" not in res:
                print(f" | Resid: {res['cv_roc_auc_mean']:.4f}", end="")

            # Within-problem
            wp = controls.get("ctrl5_within_problem", {})
            if wp and "error" not in wp:
                print(f" | Within-prob: {wp['cv_roc_auc_mean']:.4f}", end="")

            # Random
            rnd = controls.get("ctrl6_random_projection", {})
            if rnd:
                print(f" | Random: {rnd['random_auroc_mean']:.4f}", end="")

            print()  # newline

    print(f"\n{'='*90}")
    print("  INTERPRETATION GUIDE:")
    print("    • Permutation p < 0.05 → result is statistically significant")
    print("    • Pos-only AUROC high  → position confound (later steps = incorrect)")
    print("    • Len-only AUROC high  → length confound")
    print("    • Resid ≈ Real AUROC   → signal survives after removing position")
    print("    • Within-prob ≈ Real   → signal is not just problem identity")
    print("    • Random ≈ 0.5         → learned direction is special (good)")
    print(f"{'='*90}\n")


# ==================================================================
# Main
# ==================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]

    vectors_data = torch.load(args.vectors_file, weights_only=False)
    per_sample_mask = vectors_data["metadata"]["per_sample_is_fully_correct"]

    all_results = {}

    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'='*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'='*70}")

        layer_results = {}

        for gran in args.granularities:
            print(f"\n  --- {gran.upper()} level controls ---")

            if gran == "step":
                layer_results[gran] = run_controls_step_level(
                    raw_dir, hook_name, num_shards, d_model, args)
            elif gran == "sample":
                layer_results[gran] = run_controls_sample_level(
                    raw_dir, hook_name, num_shards, d_model, per_sample_mask, args)
            elif gran == "token":
                layer_results[gran] = run_controls_token_level(
                    raw_dir, hook_name, num_shards, args)

            gc.collect()

        all_results[str(layer)] = layer_results

    # Save
    out_path = os.path.join(args.output_dir, "control_baselines_results.json")

    # Convert numpy arrays in results to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    serializable = make_serializable(all_results)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()