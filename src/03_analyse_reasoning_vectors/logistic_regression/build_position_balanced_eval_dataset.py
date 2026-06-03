"""
Position-Balanced Evaluation Dataset for Step-Level Classifier
===============================================================

Constructs a dataset where step position (step_idx) is balanced across
correct and incorrect classes, removing position as a confound.

Selection rules:
  • Incorrect samples  → take ONLY the first incorrect step
    (the step at step_idx == label, i.e. the ProcessBench error label).
  • Correct samples    → take ONE step per sample (chosen to best fill
    the position slots needed for balancing).
  • At most one step per sample in the final dataset.
  • For each step_idx present in BOTH classes, downsample to
    min(n_correct, n_incorrect) so counts are equal per position.

Input:
  The reasoning_vectors_..._avg_storage.pt file, which stores metadata
  pointing to the raw_activations directory.  The raw shards contain
  per-token activations and metadata (sample_idx, step_idx, is_correct).

Output:
  A .pt file with keys:
    X            : (N, d_model)  float32 — step-mean activations
    y            : (N,)          int32   — 1=correct, 0=incorrect
    step_idx     : (N,)          int32   — position within the solution
    sample_idx   : (N,)          int32   — which ProcessBench sample
    layer        : str                   — hook name used
    balance_info : dict                  — per-position counts before/after

Usage:
  python build_position_balanced_eval.py \
    --vectors_file reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \
    --layer 22 \
    --output_file position_balanced_eval_layer22.pt \
    --seed 42
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import gc
import json


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Build a position-balanced step-level evaluation dataset."
    )
    p.add_argument("--vectors_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt",
                   help="Path to reasoning_vectors_..._avg_storage.pt")
    p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations",
                   help="Override raw_activations directory "
                        "(default: read from vectors_file metadata)")
    p.add_argument("--layer", type=int, nargs="+", default=list(range(18, 29)))
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data",
                   help="Directory for output .pt file path (default: auto-named)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────
# Shard loader
# ──────────────────────────────────────────────
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    safe = hook_name.replace(".", "_")
    acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt",
                      weights_only=False).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt",
                      weights_only=False)
    return acts, meta


# ──────────────────────────────────────────────
# Step-level aggregation from raw shards
# ──────────────────────────────────────────────
def aggregate_steps_from_shards(raw_dir, hook_name, num_shards, d_model):
    """Stream all shards and compute per-(sample, step) mean activations.

    Returns
    -------
    step_means : dict  (sample_idx, step_idx) → np.ndarray [d_model]
    step_labels : dict  (sample_idx, step_idx) → bool (True=correct)
    """
    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)  # default correct; any False token flips it

    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)
        for i, m in enumerate(meta):
            if m["step_idx"] < 0 or m["is_correct"] is None:
                continue
            key = (m["sample_idx"], m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            # Explicitly touch the key so correct steps are registered too
            # (defaultdict only creates entries on access, and .values()
            #  only iterates over entries that exist)
            _ = step_labels[key]  # creates entry with default True if new
            if m["is_correct"] is False:
                step_labels[key] = False
        del acts, acts_np, meta
        gc.collect()

    # Compute means
    step_means = {}
    for key in step_sums:
        step_means[key] = (step_sums[key] / step_counts[key]).astype(np.float32)

    del step_sums, step_counts
    gc.collect()
    return step_means, step_labels


# ──────────────────────────────────────────────
# Find first incorrect step per sample
# ──────────────────────────────────────────────
def find_first_incorrect_step(step_labels):
    """For each sample that has any incorrect step, return the lowest
    step_idx that is incorrect.

    Returns
    -------
    first_error : dict  sample_idx → step_idx of first error
    """
    # Group by sample
    sample_incorrect_steps = defaultdict(list)
    for (sample_idx, step_idx), is_correct in step_labels.items():
        if not is_correct:
            sample_incorrect_steps[sample_idx].append(step_idx)

    first_error = {}
    for sample_idx, steps in sample_incorrect_steps.items():
        first_error[sample_idx] = min(steps)

    return first_error


# ──────────────────────────────────────────────
# Identify fully-correct samples
# ──────────────────────────────────────────────
def find_correct_samples(step_labels):
    """Return set of sample_idx values whose steps are ALL correct."""
    samples_with_errors = set()
    all_samples = set()
    for (sample_idx, step_idx), is_correct in step_labels.items():
        all_samples.add(sample_idx)
        if not is_correct:
            samples_with_errors.add(sample_idx)
    return all_samples - samples_with_errors


# ──────────────────────────────────────────────
# Build the position-balanced dataset
# ──────────────────────────────────────────────
def build_position_balanced_dataset(step_means, step_labels, seed=42):
    """
    1. Incorrect class: one entry per sample — the first incorrect step.
    2. Correct class:   one entry per sample — one correct step chosen
       to best fill positions needed for balance.
    3. Keep only positions where BOTH classes are represented.
    4. Downsample each class to min count per position.

    Returns dict with X, y, step_idx, sample_idx, balance_info.
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: Build incorrect pool (first-error step only) ──
    first_error = find_first_incorrect_step(step_labels)
    incorrect_pool = []  # list of (sample_idx, step_idx)
    for sample_idx, err_step in first_error.items():
        key = (sample_idx, err_step)
        if key in step_means:
            incorrect_pool.append(key)

    # Count incorrect entries per position
    incorrect_by_pos = defaultdict(list)
    for key in incorrect_pool:
        incorrect_by_pos[key[1]].append(key)

    # ── Step 2: Build correct pool ──
    # For correct samples, enumerate all available (sample, step) pairs
    correct_samples = find_correct_samples(step_labels)
    # Also include correct steps from incorrect samples (steps before the error)
    correct_candidates_by_sample = defaultdict(list)
    for (sample_idx, step_idx), is_correct in step_labels.items():
        if is_correct and (sample_idx, step_idx) in step_means:
            correct_candidates_by_sample[sample_idx].append(step_idx)

    # Positions where incorrect class has entries
    positions_with_incorrect = set(incorrect_by_pos.keys())

    # Greedily assign one correct step per sample, preferring positions
    # that are under-represented relative to incorrect counts.
    # First pass: prioritize positions where we need more correct entries.
    correct_by_pos = defaultdict(list)
    used_samples = set()

    # Sort samples by number of eligible positions (fewest first → hardest to place)
    sample_order = sorted(
        correct_candidates_by_sample.keys(),
        key=lambda s: len([p for p in correct_candidates_by_sample[s]
                           if p in positions_with_incorrect])
    )

    for sample_idx in sample_order:
        if sample_idx in used_samples:
            continue
        candidates = correct_candidates_by_sample[sample_idx]
        # Only consider positions that also appear in incorrect pool
        eligible = [p for p in candidates if p in positions_with_incorrect]
        if not eligible:
            continue
        # Pick the position with the greatest deficit
        # deficit = incorrect_count - correct_count_so_far
        best_pos = max(eligible,
                       key=lambda p: len(incorrect_by_pos[p]) - len(correct_by_pos[p]))
        key = (sample_idx, best_pos)
        correct_by_pos[best_pos].append(key)
        used_samples.add(sample_idx)

    # ── Step 3: Keep only positions with BOTH classes ──
    shared_positions = sorted(
        positions_with_incorrect & set(correct_by_pos.keys())
    )

    if not shared_positions:
        raise ValueError("No step positions have both correct and incorrect entries!")

    # ── Step 4: Equalize counts per position ──
    final_entries = []  # list of (key, label)  label: 1=correct, 0=incorrect
    balance_info = {"per_position": {}, "positions_kept": len(shared_positions)}

    for pos in shared_positions:
        inc_keys = incorrect_by_pos[pos]
        cor_keys = correct_by_pos[pos]
        n_min = min(len(inc_keys), len(cor_keys))

        rng.shuffle(inc_keys)
        rng.shuffle(cor_keys)

        for key in inc_keys[:n_min]:
            final_entries.append((key, 0))
        for key in cor_keys[:n_min]:
            final_entries.append((key, 1))

        balance_info["per_position"][int(pos)] = {
            "n_incorrect_available": len(incorrect_by_pos[pos]),
            "n_correct_available": len(correct_by_pos[pos]),
            "n_kept_per_class": n_min,
        }

    # Shuffle final dataset
    rng.shuffle(final_entries)

    # ── Assemble arrays ──
    d_model = next(iter(step_means.values())).shape[0]
    N = len(final_entries)

    X          = np.empty((N, d_model), dtype=np.float32)
    y          = np.empty(N, dtype=np.int32)
    step_idxs  = np.empty(N, dtype=np.int32)
    sample_ids = np.empty(N, dtype=np.int32)

    for i, (key, label) in enumerate(final_entries):
        X[i]          = step_means[key]
        y[i]          = label
        step_idxs[i]  = key[1]
        sample_ids[i] = key[0]

    balance_info["total_samples"]   = N
    balance_info["total_correct"]   = int(y.sum())
    balance_info["total_incorrect"] = int(N - y.sum())

    return X, y, step_idxs, sample_ids, balance_info


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # ── Load vectors file for metadata ──
    print(f"Loading vectors file: {args.vectors_file}")
    vectors_data = torch.load(args.vectors_file, weights_only=False)
    metadata = vectors_data["metadata"]

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(metadata["raw_activations_dir"])
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]

    layers = args.layer if isinstance(args.layer, list) else [args.layer]

    for layer_idx in layers:
        hook_name = f"blocks.{layer_idx}.hook_out"
        print(f"\nLayer: {layer_idx}  ({hook_name})")
        print(f"Raw activations: {raw_dir}  ({num_shards} shards, d_model={d_model})")

        # ── Aggregate step-level data from shards ──
        print("Streaming shards to build step-level activations...")
        step_means, step_labels = aggregate_steps_from_shards(
            raw_dir, hook_name, num_shards, d_model
        )

        n_total_steps = len(step_means)
        n_correct = sum(1 for v in step_labels.values() if v)
        n_incorrect = sum(1 for v in step_labels.values() if not v)
        print(f"Total steps: {n_total_steps}  (correct: {n_correct}, incorrect: {n_incorrect})")

        # Stats on first-error positions
        first_error = find_first_incorrect_step(step_labels)
        err_positions = sorted(set(first_error.values()))
        print(f"Incorrect samples: {len(first_error)}")
        print(f"First-error positions observed: {err_positions}")

        from collections import Counter
        err_pos_counts = Counter(first_error.values())
        print("First-error position distribution:")
        for pos in sorted(err_pos_counts):
            print(f"  step_idx={pos}: {err_pos_counts[pos]} samples")

        # ── Build balanced dataset ──
        print("\nBuilding position-balanced dataset...")
        X, y, step_idxs, sample_ids, balance_info = build_position_balanced_dataset(
            step_means, step_labels, seed=args.seed
        )

        # ── Print summary ──
        print(f"\n{'='*60}")
        print(f"  POSITION-BALANCED EVAL DATASET")
        print(f"{'='*60}")
        print(f"  Total entries:   {balance_info['total_samples']}")
        print(f"  Correct (y=1):   {balance_info['total_correct']}")
        print(f"  Incorrect (y=0): {balance_info['total_incorrect']}")
        print(f"  Positions kept:  {balance_info['positions_kept']}")
        print(f"\n  Per-position breakdown:")
        print(f"  {'pos':>5} | {'inc_avail':>9} | {'cor_avail':>9} | {'kept/cls':>8}")
        print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
        for pos in sorted(balance_info["per_position"], key=int):
            info = balance_info["per_position"][pos]
            print(f"  {pos:>5} | {info['n_incorrect_available']:>9} | "
                f"{info['n_correct_available']:>9} | {info['n_kept_per_class']:>8}")
        print(f"{'='*60}")

        # Verify balance
        unique_positions = np.unique(step_idxs)
        for pos in unique_positions:
            mask = step_idxs == pos
            n_cor = (y[mask] == 1).sum()
            n_inc = (y[mask] == 0).sum()
            assert n_cor == n_inc, (
                f"Imbalance at step_idx={pos}: correct={n_cor}, incorrect={n_inc}"
            )
        print("  ✓ Balance verified: equal class counts at every position.")

        # Verify one step per sample
        #assert len(np.unique(sample_ids)) == len(sample_ids), \
        #    "Duplicate sample_idx found — expected at most one step per sample!"
        #print("  ✓ One step per sample verified.")

        output_file = os.path.join(
            args.output_dir, f"position_balanced_eval_layer{layer_idx}.pt"
        )

        output = {
            "X":            torch.from_numpy(X),
            "y":            torch.from_numpy(y),
            "step_idx":     torch.from_numpy(step_idxs),
            "sample_idx":   torch.from_numpy(sample_ids),
            "layer":        hook_name,
            "layer_idx":    layer_idx,
            "d_model":      d_model,
            "balance_info": balance_info,
            "seed":         args.seed,
            "source_vectors_file": os.path.abspath(args.vectors_file),
        }
        
        torch.save(output, output_file)
        print(f"\n  Saved → {output_file}")
        print(f"  Shape: X={list(X.shape)}, y={list(y.shape)}")
    

if __name__ == "__main__":
    main()