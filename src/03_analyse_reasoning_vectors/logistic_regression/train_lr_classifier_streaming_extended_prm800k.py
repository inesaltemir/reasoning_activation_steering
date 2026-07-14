"""
Logistic regression classifier on raw activations (residual stream)

Streams through activation shards so that at most ONE shard (≈ 1.6 GB) is in RAM at a time.

Three granularity levels:
  Token-level  → SGDClassifier(log_loss) with partial_fit, shard by shard
  Step-level   → streaming aggregation into (num_steps, d_model), then LR
  Sample-level → streaming aggregation into (num_samples, d_model), then LR

Qwen3-8B hidden dimension:   "hidden_size": 4096

Important parameters to justify / sweep over:

1. Regularization strength (--C)
    the current value is: 1.0
    should perform a sweep, with a logarithmic grid ([0.001, 0.01, 0.1, 1.0, 10.0])
    C is the inverse of the L2 regularization strength. 
    if C is not small (tough) enough, prone to overfitting, as easy to find separating hyperplane for your data points in a high-dim space
    for step and sample level data points, need stronger regularisation (smaller C), as have much less # data points
    if C is too large, risk of overfitting

    do separate sweep for token-level, step-level and sample-level

2. Training duration for SGD (--sgd_epochs)
    the current value is: 5
    should perform a sweep and monitor convergence
    the token-level model uses SGDClassifier with partial_fit. 
    monitor learning rate decay and validation AUROC to see whether model acc converges

3. Test/train split (--test_shards)
    the current value for the split is 4/19, so 17% of data points for the test set
    pretty common
n_folds == 5, standard value for cross-validation 


holdout set with different kind of text -- eval data
good separation from the get go

no risk of overfitting with linear model

more qualitative than quantitative
are we separating correctly on the holdout set
our expected thing is it will not qualitative search

move to more powerful non linear model
simple neural net (easy to get good level of non linearity separation power , relatively quick)

tree models not good

for PRM800K, need to stratify also at problem_id level! even if the prefix index is not the same, many prefix indexes share the problem definition
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
import warnings, gc
warnings.filterwarnings("ignore")


# ==========================================
# Args
# ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations")
    p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations")
    # p.add_argument("--vectors_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt")
    p.add_argument("--vectors_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/reasoning_vectors_Qwen3-8B_prm800k_with_steps_avg_storage.pt")
    p.add_argument("--target_layers", type=int, nargs="+",
                   #default=[18])
                    default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--granularities", type=str, nargs="+",
                   default=["step", "sample"],
                   choices=["token", "step", "sample"])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--sgd_epochs", type=int, default=5,
                   help="Number of full passes over shards for SGD token-level training")
    p.add_argument("--test_shards", type=int, default=4,
                   help="Number of shards held out for token-level testing")
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_prm800k")
    return p.parse_args()


# ==========================================
# Shard loader (one at a time)
# ==========================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    """Load a single branch shard + its metadata.  Returns float32 tensor + list[dict]."""
    safe = hook_name.replace(".", "_")
    acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt",
                      weights_only=False).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt",
                      weights_only=False)
    return acts, meta


# ==========================================
# PRM800K prefix cache loader
# ==========================================
def load_prefix_lookup(raw_dir: Path, hook_name: str, index: dict) -> dict:
    """Load all prefix shards into a lookup dict: prefix_id → (acts np.ndarray, meta list).

    Returns an empty dict when the dataset is not PRM800K (no prefix dedup).
    The caller is responsible for deleting the returned dict when done to free RAM.

    Parameters
    ----------
    raw_dir    : path to the raw_activations directory
    hook_name  : e.g. "blocks.22.hook_out"
    index      : the index.pt dict already loaded by the caller

    Returns
    -------
    prefix_lookup : dict[int, tuple[np.ndarray, list[dict]]]
                    prefix_id → (acts float32 (T_p, d_model), meta list[dict])
                    Empty dict if no prefix dedup info found.
    """
    prefix_index     = index.get("prefix_index")       # list[(start_row, num_tokens)]
    num_prefix_shards = index.get("num_prefix_shards", 0)

    if not prefix_index or num_prefix_shards == 0:
        return {}

    safe        = hook_name.replace(".", "_")
    prefix_safe = "prefix_" + safe

    # Load all prefix shards into one flat tensor
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

    prefix_acts_flat = np.concatenate(p_acts_list, axis=0) if p_acts_list else np.empty((0,))
    del p_acts_list

    # Build lookup: prefix_id (== position in prefix_index list) → (acts slice, meta slice)
    prefix_lookup: dict[int, tuple[np.ndarray, list]] = {}
    for pid, (pstart, plen) in enumerate(prefix_index):
        prefix_lookup[pid] = (
            prefix_acts_flat[pstart: pstart + plen],   # view — no copy
            p_meta_list[pstart: pstart + plen],
        )

    return prefix_lookup


# ==========================================
# TOKEN-LEVEL:  SGDClassifier + partial_fit
# ==========================================
def run_token_level(raw_dir, hook_name, num_shards, args):
    """
    Stream through shards.  Last `test_shards` shards are held out.
    Train via SGDClassifier.partial_fit over `sgd_epochs` passes.
    SGD updates weights incrementally and is notoriously sensitive to the number of passes over the data.
    can do sweep over sgd_epochs value for [1, 5, 10, 20] while monitoring the validation AUROC
    to seen when model has actually converged.
    """
    rng = np.random.RandomState(args.seed)

    # Decide train / test split by shard index
    shard_order = list(range(num_shards))
    rng.shuffle(shard_order)
    n_test = min(args.test_shards, max(1, num_shards // 5))
    test_shard_ids  = set(shard_order[:n_test])
    train_shard_ids = [s for s in range(num_shards) if s not in test_shard_ids]

    print(f"    Train shards: {len(train_shard_ids)}  |  Test shards: {n_test}")

    # --- Pass 1: Compute running mean/var for standardisation + class counts ---
    print(f"    Pass 1/{args.sgd_epochs + 2}: computing feature statistics...")
    scaler = StandardScaler()
    class_counts = {0: 0, 1: 0}
    for sid in train_shard_ids:
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
        if mask:
            scaler.partial_fit(acts[mask].numpy())
            for i in mask:
                label = 1 if meta[i]["is_correct"] else 0
                class_counts[label] += 1
        del acts, meta; gc.collect()

    # Compute balanced class weights: w_c = n_total / (n_classes * n_c)
    n_total = class_counts[0] + class_counts[1]
    class_weight_map = {
        c: n_total / (2.0 * count) for c, count in class_counts.items() if count > 0
    }
    print(f"    Class counts: neg={class_counts[0]:,}  pos={class_counts[1]:,}")
    print(f"    Balanced weights: {class_weight_map}")

    # --- Pass 2..N+1: partial_fit SGD ---
    # Alpha is defined as such to make the regularization strength of SGDClassifier match the regularization strength of LogisticRegression.
    # SGD alpha ≈ 1/(C*n) heuristic
    alpha = 1.0 / (args.C * n_total)  # Had harcoded 100_000, best to use total number of tokens n_total
    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,            # Constant that multiplies the regularization term. The higher the value, the stronger the regularization. 
        max_iter=1,           # we control epochs ourselves later on with args.sgd_epochs
        warm_start=True,
        random_state=args.seed,
    )

    classes = np.array([0, 1])
    for epoch in range(args.sgd_epochs):
        epoch_shards = train_shard_ids.copy()
        rng.shuffle(epoch_shards)
        n_seen = 0
        for sid in epoch_shards:
            acts, meta = load_one_shard(raw_dir, hook_name, sid)
            mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
            if not mask:
                del acts, meta; gc.collect()
                continue

            X_shard = scaler.transform(acts[mask].numpy())
            y_shard = np.array([1 if meta[i]["is_correct"] else 0 for i in mask])
            sample_weights = np.array([class_weight_map[yi] for yi in y_shard])
            clf.partial_fit(X_shard, y_shard, classes=classes,
                            sample_weight=sample_weights)
            n_seen += len(mask)
            del acts, meta, X_shard, y_shard, sample_weights; gc.collect()

        print(f"    Epoch {epoch+1}/{args.sgd_epochs}: trained on {n_seen:,} tokens")

    # --- Evaluate on held-out shards ---
    print(f"    Evaluating on test shards...")
    all_y_true, all_y_pred, all_y_prob, all_test_meta = [], [], [], []
    for sid in sorted(test_shard_ids):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
        if not mask:
            del acts, meta; gc.collect()
            continue

        X_test = scaler.transform(acts[mask].numpy())
        y_test = np.array([1 if meta[i]["is_correct"] else 0 for i in mask])

        all_y_true.append(y_test)
        all_y_pred.append(clf.predict(X_test))
        all_y_prob.append(clf.decision_function(X_test))
        all_test_meta.extend(
            {"sample_idx": meta[i]["sample_idx"],
             "step_idx":   meta[i].get("step_idx", -1),
             "shard_id":   sid}
            for i in mask
        )
        del acts, meta, X_test, y_test; gc.collect()

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob = np.concatenate(all_y_prob)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auroc":     float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred)),
        "n_test":    int(len(y_true)),
        "n_test_pos": int(y_true.sum()),
        "n_test_neg": int(len(y_true) - y_true.sum()),
        "n_train_shards": len(train_shard_ids),
        "n_test_shards":  n_test,
        "sgd_epochs": args.sgd_epochs,
        "class_counts": class_counts,
        "class_weight_map": {str(k): v for k, v in class_weight_map.items()},
        "intercept": float(clf.intercept_[0]),
    }

    # Bundle all token-level diagnostics for downstream analysis
    diagnostics = {
        # Predictions & scores: enable confusion matrices, PR curves, threshold tuning
        "y_true":     y_true,       # int array
        "y_pred":     y_pred,       # int array
        "y_prob":     y_prob,       # float array (decision function scores)
        # Per-token test metadata: enables per-sample / per-step error breakdowns
        "test_meta":  all_test_meta,  # list[dict] with sample_idx, step_idx, shard_id
        # Scaler statistics: describe the activation distribution per feature
        "scaler_mean": scaler.mean_.astype(np.float32),   # (d_model,)
        "scaler_var":  scaler.var_.astype(np.float32),     # (d_model,)
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
    }

    return metrics, clf, scaler, diagnostics


# ==========================================
# STEP-LEVEL:  streaming aggregation → LR
# ==========================================
# Because in the src/02_collect_activations/run_fw_pass_with_step_averaging_storage.py, during the token-LEVEL loop, tokens belonging to steps past
# first_err_idx will have an "is_correct" == False. ====> this label is later used for classifying step incorrectness here
# so effectively, steps after the first incorrect step ARE catalogued as incorrect too
# try with the opposite
# easy check == same number of erroneous steps as erroneous samples

# Here, aggregate all activ per step
# might want to try variant with top-k activating tokens?

def run_step_level(raw_dir, hook_name, num_shards, d_model, args, index: dict):
    """
    Stream all branch shards + prefix shards (PRM800K), accumulate per-step sums.

    Parameters
    ----------
    index : the already-loaded index.pt dict (avoids re-reading from disk).

    PRM800K dedup logic
    -------------------
    The on-disk format stores:
      - branch shards : branch-only tokens, meta has (sample_idx, branch_idx, step_idx, …)
      - prefix shards : prefix tokens stored ONCE per problem (sample_idx)

    For step-level analysis we want:
      - Each prefix step counted ONCE per problem (not once per branch).
        Key = (sample_idx, step_idx).  Because all branches of the same problem share
        sample_idx AND the same prefix step indices, the defaultdict key naturally
        deduplicates: prefix tokens from every branch shard that share (sample_idx,
        step_idx) would just keep accumulating — to avoid that we load prefix data
        separately and add it only once per prefix_id, using the prefix_lookup.
      - Each branch step counted once per branch.
        Key = (sample_idx, branch_idx, step_idx).  branch_idx is stored in branch
        metadata so different branches of the same problem do not collapse.

    For non-PRM800K datasets (no prefix dedup) the logic is unchanged.
    """
    has_prefix_dedup = bool(index.get("prefix_index") and index.get("num_prefix_shards", 0) > 0)

    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)

    # ------------------------------------------------------------------
    # PRM800K path: seed step_sums with prefix data first (each prefix once)
    # ------------------------------------------------------------------
    if has_prefix_dedup:
        print(f"    Loading prefix shards for step aggregation (PRM800K dedup)...")
        prefix_lookup = load_prefix_lookup(raw_dir, hook_name, index)

        for pid, (p_acts, p_meta) in prefix_lookup.items():
            # p_meta rows contain: sample_idx, step_idx, is_correct (and problem_id etc.)
            # We need the sample_idx of this prefix — take it from the first row.
            if not p_meta:
                continue
            sample_idx = p_meta[0]["sample_idx"]
            for row_i, m in enumerate(p_meta):
                s_idx = m.get("step_idx", -1)
                ic    = m.get("is_correct")
                if s_idx < 0 or ic is None:
                    continue
                key = (sample_idx, s_idx)          # no branch_idx — prefix is shared
                step_sums[key]   += p_acts[row_i].astype(np.float64)
                step_counts[key] += 1
                if ic is False:
                    step_labels[key] = False

        del prefix_lookup; gc.collect()

    # ------------------------------------------------------------------
    # Stream branch shards (or full-sequence shards for non-PRM800K)
    # ------------------------------------------------------------------
    print(f"    Streaming {num_shards} branch shards for step aggregation...")
    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)

        for i, m in enumerate(meta):
            s_idx = m.get("step_idx", -1)
            ic    = m.get("is_correct")
            if s_idx < 0 or ic is None:
                continue

            if has_prefix_dedup:
                # Branch tokens: key includes branch_idx so each branch step is
                # a distinct data point from other branches of the same problem.
                b_idx = m.get("branch_idx", 0)
                key = (m["sample_idx"], b_idx, s_idx)
            else:
                key = (m["sample_idx"], s_idx)

            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if ic is False:
                step_labels[key] = False

        del acts, acts_np, meta; gc.collect()

    # Build X, y
    keys = sorted(step_sums.keys())
    X = np.stack([step_sums[k] / step_counts[k] for k in keys]).astype(np.float32)
    y = np.array([1 if step_labels[k] else 0 for k in keys], dtype=np.int32)
    # groups = sample_idx (first element of key regardless of tuple length)
    groups = np.array([k[0] for k in keys])

    counts = np.array([step_counts[k] for k in keys], dtype=np.int32)

    print(f"    Steps: {len(keys)}  |  Positive: {y.sum()}  |  Negative: {len(y)-y.sum()}")

    del step_sums, step_counts, step_labels; gc.collect()

    # DATA LEAKAGE fix: use StratifiedGroupKFold so steps from the same sample
    # are never split across train/test folds.
    metrics, pipeline = _fit_cv_lr(X, y, args, groups=groups)

    diagnostics = {
        "X": X,           # (n_steps, d_model) aggregated feature matrix
        "y": y,           # (n_steps,) labels
        "keys": keys,     # list[tuple] — row identity (sample_idx[, branch_idx], step_idx)
        "token_counts": counts,   # tokens aggregated per step
        "has_prefix_dedup": has_prefix_dedup,
    }
    return metrics, pipeline, diagnostics


# ==========================================
# SAMPLE-LEVEL:  streaming aggregation → LR
# ==========================================
def run_sample_level(raw_dir, hook_name, num_shards, d_model,
                     per_sample_is_fully_correct, args, index: dict):
    """
    Stream all shards, accumulate per-sample (= per-sequence) sums.

    Parameters
    ----------
    index : the already-loaded index.pt dict (avoids re-reading from disk).

    PRM800K dedup logic
    -------------------
    On disk the prefix is stored once per problem (sample_idx) and each branch
    is a separate branch shard entry tagged with (sample_idx, branch_idx).

    We want N separate sequences for the N branches of a given problem, where
    each sequence is the concatenation of:
        prefix tokens  (shared, stored once)  +  branch tokens  (per-branch)

    So the accumulation key is (sample_idx, branch_idx) — one row per branch.
    The prefix contributions are reconstructed in-memory (never saved back to disk):
      for each branch token encountered in a shard, we look up the prefix for that
      problem and add it to the running sum for that (sample_idx, branch_idx) key,
      but only ONCE (tracked via a "prefix_added" set).

    The label for each (sample_idx, branch_idx) sequence is taken from
    per_sample_is_fully_correct.  That mask is built in the forward pass as a flat
    list over branches in dataset order, so its length equals the total number of
    (sample_idx, branch_idx) entries.  We use a separate `branch_sequence_labels`
    dict populated from per-branch meta to map (sample_idx, branch_idx) → label.

    For non-PRM800K datasets the behaviour is identical to before: key = sample_idx.
    """
    sample_index = index["sample_index"]
    has_prefix_dedup = bool(index.get("prefix_index") and index.get("num_prefix_shards", 0) > 0)

    if has_prefix_dedup:
        # ----------------------------------------------------------------
        # PRM800K path
        # ----------------------------------------------------------------
        print(f"    Loading prefix shards for sample aggregation (PRM800K dedup)...")
        prefix_lookup = load_prefix_lookup(raw_dir, hook_name, index)

        # Precompute prefix mean per prefix_id so we don't re-sum on every branch.
        # shape: prefix_id → np.ndarray (d_model,) float64
        prefix_means: dict[int, np.ndarray] = {}
        prefix_lengths: dict[int, int] = {}
        for pid, (p_acts, _) in prefix_lookup.items():
            if len(p_acts) > 0:
                prefix_means[pid]   = p_acts.astype(np.float64).sum(axis=0)
                prefix_lengths[pid] = len(p_acts)
        del prefix_lookup; gc.collect()

        # Accumulation keyed by (sample_idx, branch_idx)
        # Running sums start from the prefix contribution (added lazily on first
        # encounter of each branch in the shard stream).
        sample_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
        sample_counts = defaultdict(int)
        prefix_added  = set()   # set of (sample_idx, branch_idx) already seeded with prefix

        # Build a lookup from (sample_idx, branch_idx) → prefix_id using sample_index.
        # sample_index entries for PRM800K are 4-tuples:
        #   (start_row, num_branch_tokens, sample_idx, prefix_id)
        # We derive branch_idx from the order branches appear per sample_idx.
        branch_prefix_map: dict[tuple, int] = {}   # (sample_idx, branch_idx) → prefix_id
        _branch_counter: dict[int, int] = defaultdict(int)
        for entry in sample_index:
            si    = entry[2]
            pid   = entry[3]
            b_idx = _branch_counter[si]
            _branch_counter[si] += 1
            branch_prefix_map[(si, b_idx)] = pid
        del _branch_counter

        print(f"    Streaming {num_shards} branch shards for sample aggregation...")
        for sid in range(num_shards):
            acts, meta = load_one_shard(raw_dir, hook_name, sid)
            acts_np = acts.numpy().astype(np.float64)

            for i, m in enumerate(meta):
                s_idx = m["sample_idx"]
                b_idx = m.get("branch_idx", 0)
                key   = (s_idx, b_idx)

                # Seed with prefix sum on first encounter of this branch
                if key not in prefix_added:
                    pid = branch_prefix_map.get(key)
                    if pid is not None and pid in prefix_means:
                        sample_sums[key]   += prefix_means[pid]
                        sample_counts[key] += prefix_lengths[pid]
                    prefix_added.add(key)

                sample_sums[key]   += acts_np[i]
                sample_counts[key] += 1

            del acts, acts_np, meta; gc.collect()

        del prefix_means, prefix_lengths, prefix_added, branch_prefix_map; gc.collect()

        # Build X, y
        # Keys are (sample_idx, branch_idx); sort for determinism.
        keys = sorted(sample_sums.keys())
        X = np.stack([sample_sums[k] / sample_counts[k] for k in keys]).astype(np.float32)
        counts = np.array([sample_counts[k] for k in keys], dtype=np.int32)
        del sample_sums, sample_counts; gc.collect()

        # Labels: per_sample_is_fully_correct is a flat list/tensor over all branches,
        # ordered by their appearance in sample_index (i.e. the order they were stored
        # during the forward pass).  We build a mapping from (sample_idx, branch_idx)
        # to that flat position so we can look up the correct label for each sequence.

        # Reconstruct flat index: sample_index entries appear in dataset order;
        # to align with the sorted `keys`.
        _bi_counter: dict[int, int] = defaultdict(int)
        flat_index: dict[tuple, int] = {}   # (sample_idx, branch_idx) → flat position
        for flat_pos, entry in enumerate(sample_index):
            si    = entry[2]
            b_idx = _bi_counter[si]
            _bi_counter[si] += 1
            flat_index[(si, b_idx)] = flat_pos

        y = np.array([
            int(per_sample_is_fully_correct[flat_index[k]].item())
            if k in flat_index and flat_index[k] < len(per_sample_is_fully_correct)
            else -1
            for k in keys
        ], dtype=np.int32)

        # Drop rows with no label (shouldn't happen in practice)
        valid = y >= 0
        if not valid.all():
            print(f"    Warning: dropping {(~valid).sum()} branch rows with no label.")
            X      = X[valid]
            y      = y[valid]
            counts = counts[valid]
            keys   = [k for k, v in zip(keys, valid) if v]

        # groups = sample_idx for StratifiedGroupKFold (no cross-problem leakage)
        groups = np.array([k[0] for k in keys])

        print(f"    Sequences (prefix+branch): {len(y)}  |  Positive: {y.sum()}  |  Negative: {len(y)-y.sum()}")

        metrics, pipeline = _fit_cv_lr(X, y, args, groups=groups)

        diagnostics = {
            "X": X,
            "y": y,
            "sequence_keys": keys,   # list[(sample_idx, branch_idx)]
            "token_counts": counts,
            "has_prefix_dedup": True,
        }
        return metrics, pipeline, diagnostics

    else:
        # ----------------------------------------------------------------
        # Non-PRM800K path (original logic, unchanged)
        # ----------------------------------------------------------------
        sample_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
        sample_counts = defaultdict(int)

        print(f"    Streaming {num_shards} shards for sample aggregation...")
        for sid in range(num_shards):
            acts, meta = load_one_shard(raw_dir, hook_name, sid)
            acts_np = acts.numpy().astype(np.float64)

            for i, m in enumerate(meta):
                sid_sample = m["sample_idx"]
                sample_sums[sid_sample]   += acts_np[i]
                sample_counts[sid_sample] += 1

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

        print(f"    Samples: {len(y)}  |  Positive: {y.sum()}  |  Negative: {len(y)-y.sum()}")

        del sample_sums, sample_counts; gc.collect()

        metrics, pipeline = _fit_cv_lr(X, y, args)

        diagnostics = {
            "X": X,
            "y": y,
            "sample_ids": sample_ids[:len(y)],
            "token_counts": counts,
            "has_prefix_dedup": False,
        }
        return metrics, pipeline, diagnostics


# ==========================================
# Shared: CV logistic regression for step/sample
# ==========================================
def _fit_cv_lr(X, y, args, groups=None):
    """Stratified K-fold CV on a StandardScaler → LR pipeline."""
    if len(np.unique(y)) < 2:
        return {"error": "single_class"}, None

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=args.C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=args.seed)),
    ])
    # in this case, C is the inverse of the L2 regularisation strength.
    # What should be the ration between # data points and C value?
    # should do a sweep of parameters

    scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

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

    pipeline.fit(X, y)

    lr_model = pipeline.named_steps["lr"]

    metrics = {}
    for s in scoring:
        metrics[f"cv_{s}_folds"]      = cv_results[f"test_{s}"].tolist()
        metrics[f"cv_{s}_mean"]       = float(np.mean(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_std"]        = float(np.std(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_train_folds"] = cv_results[f"train_{s}"].tolist()
        metrics[f"cv_{s}_train_mean"]  = float(np.mean(cv_results[f"train_{s}"]))
        metrics[f"cv_{s}_train_std"]   = float(np.std(cv_results[f"train_{s}"]))

    metrics["n_samples"]  = int(len(y))
    metrics["n_positive"] = int(y.sum())
    metrics["n_negative"] = int(len(y) - y.sum())
    metrics["n_folds"]    = n_folds

    metrics["intercept"] = float(lr_model.intercept_[0])

    metrics["n_iter"] = int(lr_model.n_iter_[0])
    metrics["max_iter"] = 2000

    metrics["cv_fit_time_mean"]   = float(np.mean(cv_results["fit_time"]))
    metrics["cv_fit_time_std"]    = float(np.std(cv_results["fit_time"]))
    metrics["cv_score_time_mean"] = float(np.mean(cv_results["score_time"]))
    metrics["cv_score_time_std"]  = float(np.std(cv_results["score_time"]))

    cv_scaler = pipeline.named_steps["scaler"]
    metrics["scaler_mean_norm"] = float(np.linalg.norm(cv_scaler.mean_))
    metrics["scaler_var_mean"]  = float(np.mean(cv_scaler.var_))

    return metrics, pipeline


# ==========================================
# Weight vector comparison
# ==========================================
def compare_weights(pipeline_or_clf, scaler, layers_dict, hook_name):
    """Cosine similarity between LR weights and stored reasoning directions."""
    import torch.nn.functional as F

    if isinstance(pipeline_or_clf, Pipeline):
        scaler = pipeline_or_clf.named_steps["scaler"]
        coef = pipeline_or_clf.named_steps["lr"].coef_[0]
    else:
        coef = pipeline_or_clf.coef_[0]

    w_raw = coef / (scaler.scale_ + 1e-12)
    w = torch.tensor(w_raw, dtype=torch.float32)

    if hook_name not in layers_dict:
        return {}

    comps = {}
    for name, vec in layers_dict[hook_name].items():
        if "reasoning_direction" in name and isinstance(vec, torch.Tensor) and vec.dim() == 1:
            cos = F.cosine_similarity(w.unsqueeze(0), vec.float().unsqueeze(0)).item()
            comps[name] = round(cos, 4)
    return comps


# ==========================================
# Cross-granularity weight comparison
# ==========================================
def compute_cross_granularity_similarities(all_learned_weights):
    """
    Cosine similarity between every pair of (layer, granularity) weight vectors.
    Returns a nested dict: results[layer][(granA, granB)] = cosine_sim.
    """
    import torch.nn.functional as F
    cross_sims = {}

    for layer_key, gran_dict in all_learned_weights.items():
        grans = sorted(gran_dict.keys())
        layer_sims = {}
        for i, g1 in enumerate(grans):
            for g2 in grans[i + 1:]:
                w1 = gran_dict[g1].unsqueeze(0)
                w2 = gran_dict[g2].unsqueeze(0)
                cos = F.cosine_similarity(w1, w2).item()
                layer_sims[f"{g1}_vs_{g2}"] = round(cos, 4)
        cross_sims[layer_key] = layer_sims

    # Cross-layer comparison within same granularity
    layers = sorted(all_learned_weights.keys(), key=int)
    if len(layers) > 1:
        all_grans = set()
        for gd in all_learned_weights.values():
            all_grans.update(gd.keys())
        for g in sorted(all_grans):
            key = f"cross_layer_{g}"
            cross_sims[key] = {}
            for i, l1 in enumerate(layers):
                for l2 in layers[i + 1:]:
                    if g in all_learned_weights[l1] and g in all_learned_weights[l2]:
                        w1 = all_learned_weights[l1][g].unsqueeze(0)
                        w2 = all_learned_weights[l2][g].unsqueeze(0)
                        cos = F.cosine_similarity(w1, w2).item()
                        cross_sims[key][f"layer{l1}_vs_layer{l2}"] = round(cos, 4)

    return cross_sims


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]

    vectors_data = torch.load(args.vectors_file, weights_only=False)
    per_sample_mask = vectors_data["metadata"]["per_sample_is_fully_correct"]
    layers_dict = vectors_data.get("layers", {})

    all_results = {}
    all_learned_weights = {}
    all_diagnostics = {}            

    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'='*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'='*70}")

        layer_results = {}
        layer_diagnostics = {}

        for gran in args.granularities:
            print(f"\n  --- {gran.upper()} level ---")

            if gran == "token":
                metrics, clf, scaler, diagnostics = run_token_level(
                    raw_dir, hook_name, num_shards, args)
                comps = compare_weights(clf, scaler, layers_dict, hook_name)

            elif gran == "step":
                metrics, pipeline, diagnostics = run_step_level(
                    raw_dir, hook_name, num_shards, d_model, args, index)
                comps = compare_weights(pipeline, None, layers_dict, hook_name) if pipeline else {}

            elif gran == "sample":
                metrics, pipeline, diagnostics = run_sample_level(
                    raw_dir, hook_name, num_shards, d_model, per_sample_mask, args, index)
                comps = compare_weights(pipeline, None, layers_dict, hook_name) if pipeline else {}

            if isinstance(metrics, dict) and "error" not in metrics:
                if gran == "token":
                    print(f"    Accuracy : {metrics['accuracy']:.4f}")
                    print(f"    AUROC    : {metrics['auroc']:.4f}")
                    print(f"    F1       : {metrics['f1']:.4f}")
                else:
                    print(f"    CV Accuracy : {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
                    print(f"    CV AUROC    : {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
                    print(f"    CV F1       : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

                if comps:
                    metrics["weight_vs_reasoning_vectors"] = comps
                    print(f"    LR weight vs reasoning directions:")
                    for vn, cs in sorted(comps.items()):
                        print(f"      {vn:<45}: {cs:+.4f}")

            layer_results[gran] = metrics
            layer_diagnostics[gran] = diagnostics  

            # Extract weight vectors 
            if gran == "token":
                w_raw = clf.coef_[0] / (scaler.scale_ + 1e-12)
                # Project intercept into input space to match w_raw
                metrics["intercept_input_space"] = float(
                    clf.intercept_[0] - w_raw @ scaler.mean_)
            else:
                model = pipeline.named_steps["lr"]
                cv_scaler = pipeline.named_steps["scaler"]
                w_raw = model.coef_[0] / (cv_scaler.scale_ + 1e-12)
                # Project intercept into input space to match w_raw
                metrics["intercept_input_space"] = float(
                    model.intercept_[0] - w_raw @ cv_scaler.mean_)

            if str(layer) not in all_learned_weights:
                all_learned_weights[str(layer)] = {}
            all_learned_weights[str(layer)][gran] = torch.tensor(w_raw, dtype=torch.float32)

            gc.collect()

        all_results[str(layer)] = layer_results
        all_diagnostics[str(layer)] = layer_diagnostics

    # ------------------------------------------------------------------
    # Cross-granularity & cross-layer weight cosine similarities
    # ------------------------------------------------------------------
    cross_sims = compute_cross_granularity_similarities(all_learned_weights)
    all_results["cross_weight_similarities"] = cross_sims

    print(f"\n  --- Cross-granularity weight similarities ---")
    for section, pairs in cross_sims.items():
        if pairs:
            print(f"    [{section}]")
            for pair_name, cos_val in pairs.items():
                print(f"      {pair_name:<40}: {cos_val:+.4f}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # 1. JSON metrics (human-readable, all scalars + per-fold arrays)
    results_path = os.path.join(args.output_dir, "lr_classifier_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # 2. Learned weight vectors (unchanged)
    weights_path = os.path.join(args.output_dir, "lr_learned_weights.pt")
    torch.save(all_learned_weights, weights_path)
    print(f"Weights saved → {weights_path}")

    # 3. Full diagnostics bundle (one .pt file per layer)
    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    for layer_key, layer_diag in all_diagnostics.items():
        layer_path = os.path.join(diag_dir, f"layer_{layer_key}_diagnostics.pt")
        torch.save(layer_diag, layer_path)
        print(f"Diagnostics saved → {layer_path}")


if __name__ == "__main__":
    main()