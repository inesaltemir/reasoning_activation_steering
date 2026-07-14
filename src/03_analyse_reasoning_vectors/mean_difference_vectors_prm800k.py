"""
compute_mean_diff_vectors_prm800k.py
-------------------------------------
Reads the raw activations produced by
run_fw_pass_with_step_averaging_storage_v2_prm800k.py for the PRM800K
dataset and computes, for each layer:

    steering_vector = mean over all (sample_idx, prefix_id) groups of
                      mean over all (correct_branch, incorrect_branch) pairs of
                      [mean_act(correct_branch) - mean_act(incorrect_branch)]

That is:
  1. Load the branch-level index from index.pt.
  2. Group entries by (sample_idx, prefix_id) — each group is a set of branches
     that share the same problem prefix.
  3. Within each group, split branches into correct (is_correct == True) and
     incorrect (is_correct == False) using the per-token metadata.
  4. For each branch, compute its mean activation vector across all its tokens.
  5. Form all pairwise differences:  mean_act(correct) - mean_act(incorrect).
  6. Average all pairwise differences across every pair in every group
     → one steering vector per layer.

Output:  a .pt file  {output_dir}/mean_diff_vectors_prm800k.pt
    {
        "layers": {
            "blocks.18.hook_out": {
                "steering_vector":   Tensor[d_model],   # final averaged diff
                "num_pairs":         int,                # total pairs used
                "num_groups":        int,                # groups that contributed
            },
            ...
        },
        "metadata": { ... }
    }
"""

import argparse
import logging
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute pairwise correct-incorrect mean-diff steering vectors "
                    "from PRM800K raw activations."
    )
    p.add_argument(
        "--raw-activations-dir",
        default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations",
        help="Path to the raw_activations directory written by the forward-pass script "
             "(contains index.pt, meta_shard_*.pt, blocks.*/ shards, prefix_*/ shards).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save mean_diff_vectors_prm800k.pt.  "
             "Defaults to the same directory as --raw-activations-dir.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Subset of hook names to process, e.g. blocks.22.hook_out blocks.24.hook_out.  "
             "Defaults to all layers found in the index.",
    )
    p.add_argument(
        "--branch-tokens",
        choices=["branch_only", "full"],
        default="branch_only",
        help="Whether to use branch-only tokens or prefix+branch tokens when computing "
             "each branch's mean activation.  'branch_only' is faster and avoids "
             "diluting the branch signal with shared prefix tokens (default).",
    )
    p.add_argument(
        "--min-pairs-per-group",
        type=int,
        default=1,
        help="Minimum number of valid (correct, incorrect) pairs a group must have "
             "to be included.  Default: 1.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shard loader  (lazy, shard-by-shard to avoid loading everything at once)
# ---------------------------------------------------------------------------
class ShardedActivationLoader:
    """Loads branch activations one shard at a time with an offset map."""

    def __init__(self, raw_dir: Path, hook_name: str, num_shards: int):
        self.raw_dir = raw_dir
        self.safe = hook_name.replace(".", "_")
        self.num_shards = num_shards
        self._cache: dict[int, torch.Tensor] = {}   # shard_id → tensor
        self._shard_row_starts: list[int] = []       # absolute row of each shard's first row
        self._shard_lengths: list[int] = []

        # Pre-scan shard lengths so we can map absolute row → shard
        cumulative = 0
        for sid in range(num_shards):
            t = torch.load(self._shard_path(sid), weights_only=False)
            self._shard_row_starts.append(cumulative)
            self._shard_lengths.append(t.shape[0])
            cumulative += t.shape[0]
            del t

    def _shard_path(self, sid: int) -> Path:
        return self.raw_dir / self.safe / f"shard_{sid:04d}.pt"

    def _get_shard(self, sid: int) -> torch.Tensor:
        if sid not in self._cache:
            # Evict all other cached shards to keep memory low
            self._cache.clear()
            self._cache[sid] = torch.load(self._shard_path(sid), weights_only=False)
        return self._cache[sid]

    def get_rows(self, start_row: int, num_rows: int) -> torch.Tensor:
        """Return activations for rows [start_row, start_row+num_rows)."""
        # Find which shard(s) contain these rows
        parts = []
        remaining_start = start_row
        remaining_len   = num_rows
        for sid in range(self.num_shards):
            s_start = self._shard_row_starts[sid]
            s_len   = self._shard_lengths[sid]
            s_end   = s_start + s_len
            if remaining_start >= s_end:
                continue
            if remaining_start + remaining_len <= s_start:
                break
            local_start = remaining_start - s_start
            local_end   = min(local_start + remaining_len, s_len)
            shard = self._get_shard(sid)
            parts.append(shard[local_start:local_end])
            taken = local_end - local_start
            remaining_start += taken
            remaining_len   -= taken
            if remaining_len <= 0:
                break
        return torch.cat(parts, dim=0) if parts else torch.empty(0)


# ---------------------------------------------------------------------------
# Prefix loader  (small enough to load fully into memory once)
# ---------------------------------------------------------------------------
def load_prefix_activations(raw_dir: Path, hook_name: str,
                             num_prefix_shards: int) -> torch.Tensor:
    safe = "prefix_" + hook_name.replace(".", "_")
    parts = []
    for sid in range(num_prefix_shards):
        parts.append(torch.load(
            raw_dir / safe / f"shard_{sid:04d}.pt", weights_only=False))
    return torch.cat(parts, dim=0) if parts else torch.empty(0)


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------
def load_all_meta(raw_dir: Path, num_shards: int) -> list[dict]:
    meta = []
    for sid in range(num_shards):
        meta.extend(torch.load(
            raw_dir / f"meta_shard_{sid:04d}.pt", weights_only=False))
    return meta


def load_all_prefix_meta(raw_dir: Path, num_prefix_shards: int) -> list[dict]:
    meta = []
    for sid in range(num_prefix_shards):
        meta.extend(torch.load(
            raw_dir / "prefix_meta" / f"shard_{sid:04d}.pt", weights_only=False))
    return meta


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_branch_mean(
    loader: ShardedActivationLoader,
    start_row: int,
    num_rows: int,
    prefix_acts: torch.Tensor | None,
    prefix_start: int,
    prefix_len: int,
    include_prefix: bool,
) -> torch.Tensor | None:
    """Return mean activation vector for one branch.

    If include_prefix is True, concatenates the shared prefix tokens.
    Returns None if the branch has 0 tokens.
    """
    branch_acts = loader.get_rows(start_row, num_rows)   # (T_branch, d_model)
    if include_prefix and prefix_acts is not None and prefix_len > 0:
        p_acts = prefix_acts[prefix_start: prefix_start + prefix_len]
        acts = torch.cat([p_acts, branch_acts], dim=0)
    else:
        acts = branch_acts
    if acts.shape[0] == 0:
        return None
    return acts.float().mean(dim=0)   # (d_model,)


def process_layer(
    raw_dir: Path,
    hook_name: str,
    index: dict,
    all_branch_meta: list[dict],
    all_prefix_meta: list[dict],
    include_prefix: bool,
    min_pairs: int,
) -> dict:
    """Compute the mean-diff steering vector for one layer.

    Returns a dict with keys: steering_vector, num_pairs, num_groups.
    """
    num_shards        = index["num_shards"]
    num_prefix_shards = index.get("num_prefix_shards", 0)
    sample_index      = index["sample_index"]
    prefix_index      = index.get("prefix_index") or []

    has_prefix_dedup = (len(prefix_index) > 0 and num_prefix_shards > 0)

    # Load prefix activations once (if needed)
    prefix_acts_flat: torch.Tensor | None = None
    if has_prefix_dedup and include_prefix:
        prefix_acts_flat = load_prefix_activations(raw_dir, hook_name, num_prefix_shards)

    # Lazy branch shard loader
    loader = ShardedActivationLoader(raw_dir, hook_name, num_shards)

    # -----------------------------------------------------------------------
    # Group sample-index entries by (sample_idx, prefix_id)
    # Each entry for PRM800K is a 4-tuple: (start_row, num_tokens, sample_idx, prefix_id)
    # -----------------------------------------------------------------------
    groups: dict[tuple, list[int]] = defaultdict(list)
    for entry_idx, entry in enumerate(sample_index):
        if len(entry) == 4:
            start_row, num_tokens, sample_idx, prefix_id = entry
        else:
            # Non-dedup fallback (shouldn't happen for PRM800K but handle gracefully)
            start_row, num_tokens, sample_idx = entry[:3]
            prefix_id = -1
        groups[(sample_idx, prefix_id)].append(entry_idx)

    # -----------------------------------------------------------------------
    # For each group, split branches into correct / incorrect using metadata
    # then compute all pairwise differences
    # -----------------------------------------------------------------------
    running_diff_sum = None   # will be a (d_model,) float32 tensor
    total_pairs  = 0
    total_groups = 0

    for (sample_idx, prefix_id), entry_indices in tqdm(
        groups.items(), desc=f"  groups [{hook_name}]", leave=False
    ):
        # Resolve prefix activation slice (shared across all branches in group)
        p_start, p_len = 0, 0
        if has_prefix_dedup and 0 <= prefix_id < len(prefix_index):
            p_start, p_len = prefix_index[prefix_id]

        correct_means   = []
        incorrect_means = []

        for entry_idx in entry_indices:
            entry = sample_index[entry_idx]
            if len(entry) == 4:
                start_row, num_tokens, _, _ = entry
            else:
                start_row, num_tokens = entry[0], entry[1]

            # Determine branch correctness from its metadata rows
            # We look at the *branch* metadata rows aligned to this entry.
            # branch_meta_rows are in all_branch_meta at positions [start_row, start_row+num_tokens)
            branch_meta_slice = all_branch_meta[start_row: start_row + num_tokens]

            # A branch is "correct" if its leading completion token is correct,
            # i.e., the first token's is_correct in the branch metadata.
            # We use majority vote across the branch's tokens to be robust.
            correct_votes   = sum(1 for m in branch_meta_slice if m.get("is_correct") is True)
            incorrect_votes = sum(1 for m in branch_meta_slice if m.get("is_correct") is False)
            if correct_votes == 0 and incorrect_votes == 0:
                continue  # no label information → skip
            branch_is_correct = correct_votes >= incorrect_votes

            # Compute mean activation for this branch
            mean_act = compute_branch_mean(
                loader,
                start_row=start_row,
                num_rows=num_tokens,
                prefix_acts=prefix_acts_flat,
                prefix_start=p_start,
                prefix_len=p_len,
                include_prefix=include_prefix,
            )
            if mean_act is None:
                continue

            if branch_is_correct:
                correct_means.append(mean_act)
            else:
                incorrect_means.append(mean_act)

        if len(correct_means) == 0 or len(incorrect_means) == 0:
            continue  # group has no contrast pairs

        # All pairwise differences: (correct, incorrect)
        group_diffs = []
        for c_vec, i_vec in product(correct_means, incorrect_means):
            group_diffs.append(c_vec - i_vec)

        n_pairs = len(group_diffs)
        if n_pairs < min_pairs:
            continue

        group_mean_diff = torch.stack(group_diffs, dim=0).mean(dim=0)  # (d_model,)

        if running_diff_sum is None:
            running_diff_sum = group_mean_diff.clone()
        else:
            running_diff_sum += group_mean_diff

        total_pairs  += n_pairs
        total_groups += 1

    if running_diff_sum is None or total_groups == 0:
        d_model = index["d_model"]
        steering_vector = torch.zeros(d_model, dtype=torch.float32)
    else:
        steering_vector = running_diff_sum / total_groups

    return {
        "steering_vector": steering_vector,
        "num_pairs":       total_pairs,
        "num_groups":      total_groups,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    raw_dir    = Path(args.raw_activations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else raw_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load index
    index_path = raw_dir / "index.pt"
    if not index_path.exists():
        raise FileNotFoundError(f"index.pt not found at {index_path}.")
    index = torch.load(index_path, weights_only=False)
    logging.info(f"Loaded index: {index['total_token_rows']} branch token rows, "
                 f"{index['num_shards']} shards.")

    all_hook_names: list[str] = index.get("hook_names", [])
    if args.layers:
        hook_names = [h for h in args.layers if h in all_hook_names]
        missing = [h for h in args.layers if h not in all_hook_names]
        if missing:
            logging.warning(f"Requested layers not found in index: {missing}")
    else:
        hook_names = all_hook_names

    if not hook_names:
        raise ValueError("No valid hook names to process.")
    logging.info(f"Processing layers: {hook_names}")

    include_prefix = (args.branch_tokens == "full")
    logging.info(f"Branch representation: {'prefix + branch tokens' if include_prefix else 'branch tokens only'}")

    # Load all branch metadata once (needed to determine branch correctness)
    logging.info("Loading branch metadata …")
    all_branch_meta = load_all_meta(raw_dir, index["num_shards"])
    logging.info(f"  {len(all_branch_meta)} branch-token metadata rows loaded.")

    has_prefix = (index.get("prefix_index") and index.get("num_prefix_shards", 0) > 0)
    all_prefix_meta: list[dict] = []
    if has_prefix:
        logging.info("Loading prefix metadata …")
        all_prefix_meta = load_all_prefix_meta(raw_dir, index["num_prefix_shards"])
        logging.info(f"  {len(all_prefix_meta)} prefix-token metadata rows loaded.")

    # Process each layer
    results: dict = {"layers": {}, "metadata": {}}
    for hook_name in hook_names:
        logging.info(f"\nProcessing layer: {hook_name}")
        layer_result = process_layer(
            raw_dir=raw_dir,
            hook_name=hook_name,
            index=index,
            all_branch_meta=all_branch_meta,
            all_prefix_meta=all_prefix_meta,
            include_prefix=include_prefix,
            min_pairs=args.min_pairs_per_group,
        )
        results["layers"][hook_name] = layer_result
        logging.info(
            f"  → groups: {layer_result['num_groups']}, "
            f"pairs: {layer_result['num_pairs']}, "
            f"steering vector norm: {layer_result['steering_vector'].norm():.4f}"
        )

    results["metadata"] = {
        "raw_activations_dir": str(raw_dir),
        "hook_names":          hook_names,
        "branch_tokens":       args.branch_tokens,
        "min_pairs_per_group": args.min_pairs_per_group,
        "index_total_rows":    index["total_token_rows"],
    }

    out_path = output_dir / "mean_diff_vectors_prm800k.pt"
    torch.save(results, out_path)
    logging.info(f"\nSaved steering vectors → {out_path}")

    # Quick summary
    print("\n=== Summary ===")
    for hook_name, res in results["layers"].items():
        print(f"  {hook_name}: "
              f"groups={res['num_groups']}, "
              f"pairs={res['num_pairs']}, "
              f"||v||={res['steering_vector'].norm():.4f}")


if __name__ == "__main__":
    main()