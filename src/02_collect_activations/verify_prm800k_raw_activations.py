"""
verify_prm800k_raw_activations.py
──────────────────────────────────
Quick sanity-check and reconstruction demo for a PRM800K raw_activations/ folder.

Usage:
    python verify_prm800k_raw_activations.py \
        --raw_dir /home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations \
        --sample_idx 0 \
        --layer 22

What it does (in order):
  1. STRUCTURE CHECK  – lists every folder / file in raw_activations/ and checks
                        that all expected layer dirs (branch + prefix) exist and
                        have the same shard count.
  2. INDEX INSPECTION – prints every field of index.pt and shows the first few
                        sample_index and prefix_index entries.
  3. SHARD SHAPES     – loads shard_0000.pt from one branch layer and one prefix
                        layer; verifies shapes and dtypes.
  4. META ALIGNMENT   – loads meta_shard_0000.pt and prefix_meta/shard_0000.pt;
                        verifies row counts match the corresponding act shards;
                        prints a few example dicts.
  5. RECONSTRUCTION   – for a chosen sample_idx, finds all branches that share
                        the same prefix_id, then for EACH branch:
                           a. loads prefix token-ids from prefix_meta
                           b. loads branch token-ids from branch meta
                           c. detokenizes with the Qwen3 tokenizer
                           d. prints the decoded text with colour-coded
                              step labels (✓ / ✗ / ?)
  6. QUICK STATS      – counts correct / incorrect / None tokens across shard 0
                        and prints the class balance.
"""

import os, sys, argparse, textwrap
from pathlib import Path
from collections import defaultdict

import torch

# ── colour helpers (ANSI, graceful fallback if piped) ────────────────────────
def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

green  = lambda t: _c("32", t)
red    = lambda t: _c("31", t)
yellow = lambda t: _c("33", t)
bold   = lambda t: _c("1",  t)
dim    = lambda t: _c("2",  t)

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",    default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/prm800k/raw_activations",
                   help="Path to raw_activations/ directory")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Dataset row index to reconstruct (default: 0)")
    p.add_argument("--layer",      type=int, default=22,
                   help="Which layer to inspect activations for (default: 22)")
    p.add_argument("--model_name", default="Qwen/Qwen3-8B",
                   help="HuggingFace model name for tokenizer (default: Qwen/Qwen3-8B)")
    p.add_argument("--no_tokenizer", action="store_true",
                   help="Skip tokenizer loading (step 5 will show token-ids only)")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – STRUCTURE CHECK
# ═════════════════════════════════════════════════════════════════════════════
def check_structure(raw_dir: Path, index: dict):
    print(bold("\n══ STEP 1: Directory structure ══"))

    hook_names = index["hook_names"]  # e.g. ["blocks.18.hook_out", ..., "blocks.28.hook_out"]
    num_shards        = index["num_shards"]
    num_prefix_shards = index.get("num_prefix_shards", 0)
    has_prefix        = (index.get("prefix_index") is not None and num_prefix_shards > 0)

    print(f"  Layers stored : {hook_names}")
    print(f"  Branch shards : {num_shards}")
    print(f"  Prefix shards : {num_prefix_shards}  (prm800k dedup={'YES' if has_prefix else 'NO'})")
    print()

    ok = True
    for name in hook_names:
        safe = name.replace(".", "_")

        # branch dir
        bdir = raw_dir / safe
        if not bdir.exists():
            print(red(f"  MISSING branch dir: {bdir}"))
            ok = False
        else:
            found = len(list(bdir.glob("shard_*.pt")))
            status = green("✓") if found == num_shards else red(f"✗ expected {num_shards}")
            print(f"  {safe}/   {found} shards  {status}")

        if has_prefix:
            pdir = raw_dir / ("prefix_" + safe)
            if not pdir.exists():
                print(red(f"  MISSING prefix dir: {pdir}"))
                ok = False
            else:
                found = len(list(pdir.glob("shard_*.pt")))
                status = green("✓") if found == num_prefix_shards else red(f"✗ expected {num_prefix_shards}")
                print(f"  prefix_{safe}/   {found} shards  {status}")

    # meta files
    branch_metas = sorted(raw_dir.glob("meta_shard_*.pt"))
    print(f"\n  meta_shard_*.pt files : {len(branch_metas)}  ", end="")
    print(green("✓") if len(branch_metas) == num_shards else red(f"✗ expected {num_shards}"))

    if has_prefix:
        pmeta_dir = raw_dir / "prefix_meta"
        if not pmeta_dir.exists():
            print(red("  MISSING prefix_meta/ directory"))
            ok = False
        else:
            pmetas = sorted(pmeta_dir.glob("shard_*.pt"))
            print(f"  prefix_meta/ shard files : {len(pmetas)}  ", end="")
            print(green("✓") if len(pmetas) == num_prefix_shards else red(f"✗ expected {num_prefix_shards}"))

    idx_path = raw_dir / "index.pt"
    print(f"  index.pt exists : ", green("✓") if idx_path.exists() else red("✗"))
    print()
    return ok


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – INDEX INSPECTION
# ═════════════════════════════════════════════════════════════════════════════
def inspect_index(index: dict):
    print(bold("══ STEP 2: index.pt contents ══"))

    scalar_fields = ["total_token_rows", "num_shards", "d_model", "dtype",
                     "num_prefix_shards"]
    for f in scalar_fields:
        print(f"  {f:30s} = {index.get(f)}")

    si = index["sample_index"]
    pi = index.get("prefix_index")
    print(f"  {'sample_index length':30s} = {len(si)}  entries")
    if pi:
        print(f"  {'prefix_index length':30s} = {len(pi)}  entries")

    print(f"\n  hook_names: {index['hook_names']}")

    print(f"\n  First 5 sample_index entries (start_row, n_tokens, sample_idx, prefix_id):")
    for e in si[:5]:
        print(f"    {e}")

    if pi:
        print(f"\n  First 5 prefix_index entries (start_row, n_tokens):")
        for e in pi[:5]:
            print(f"    {e}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 – SHARD SHAPES
# ═════════════════════════════════════════════════════════════════════════════
def check_shard_shapes(raw_dir: Path, index: dict, layer: int):
    print(bold(f"══ STEP 3: Shard shapes (layer {layer}) ══"))

    hook_name = f"blocks.{layer}.hook_out"
    safe      = hook_name.replace(".", "_")
    d_model   = index["d_model"]

    branch_shard = raw_dir / safe / "shard_0000.pt"
    if branch_shard.exists():
        t = torch.load(branch_shard, weights_only=False)
        match = green("✓") if t.shape[1] == d_model else red("✗ d_model mismatch")
        print(f"  Branch shard_0000.pt : shape={tuple(t.shape)}  dtype={t.dtype}  {match}")
    else:
        print(red(f"  Branch shard_0000.pt not found: {branch_shard}"))

    prefix_shard = raw_dir / ("prefix_" + safe) / "shard_0000.pt"
    if prefix_shard.exists():
        t = torch.load(prefix_shard, weights_only=False)
        match = green("✓") if t.shape[1] == d_model else red("✗ d_model mismatch")
        print(f"  Prefix shard_0000.pt : shape={tuple(t.shape)}  dtype={t.dtype}  {match}")
    else:
        print(yellow(f"  Prefix shard_0000.pt not found (expected for prm800k)"))
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 – METADATA ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════════
def check_meta_alignment(raw_dir: Path, index: dict, layer: int):
    print(bold(f"══ STEP 4: Meta alignment check (shard 0, layer {layer}) ══"))

    safe = f"blocks_{layer}_hook_out"

    # Branch
    branch_act_path  = raw_dir / safe / "shard_0000.pt"
    branch_meta_path = raw_dir / "meta_shard_0000.pt"
    if branch_act_path.exists() and branch_meta_path.exists():
        acts = torch.load(branch_act_path, weights_only=False)
        meta = torch.load(branch_meta_path, weights_only=False)
        match = green("✓") if len(meta) == acts.shape[0] else red(f"✗  acts={acts.shape[0]}  meta={len(meta)}")
        print(f"  Branch shard 0: acts={acts.shape[0]} rows, meta={len(meta)} rows  {match}")
        print(f"  Example branch meta rows:")
        for m in meta[:3]:
            print(f"    {m}")

    # Prefix
    pmeta_path = raw_dir / "prefix_meta" / "shard_0000.pt"
    pact_path  = raw_dir / ("prefix_" + safe) / "shard_0000.pt"
    if pact_path.exists() and pmeta_path.exists():
        pacts = torch.load(pact_path, weights_only=False)
        pmeta = torch.load(pmeta_path, weights_only=False)
        match = green("✓") if len(pmeta) == pacts.shape[0] else red(f"✗  acts={pacts.shape[0]}  meta={len(pmeta)}")
        print(f"\n  Prefix shard 0: acts={pacts.shape[0]} rows, meta={len(pmeta)} rows  {match}")
        print(f"  Example prefix meta rows:")
        for m in pmeta[:3]:
            print(f"    {m}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 – RECONSTRUCTION + DETOKENIZATION
# ═════════════════════════════════════════════════════════════════════════════
def reconstruct_and_decode(raw_dir: Path, index: dict, target_sample_idx: int,
                           tokenizer=None):
    print(bold(f"══ STEP 5: Reconstruct sample_idx={target_sample_idx} ══"))

    si             = index["sample_index"]
    pi             = index.get("prefix_index")
    num_shards     = index["num_shards"]
    num_p_shards   = index.get("num_prefix_shards", 0)
    has_prefix     = (pi is not None and num_p_shards > 0)

    # ── find all branches for this sample ──────────────────────────────────
    branches_for_sample = [e for e in si if e[2] == target_sample_idx]
    if not branches_for_sample:
        print(red(f"  No entries found for sample_idx={target_sample_idx}"))
        print(f"  Valid sample_idx values: 0 … {max(e[2] for e in si)}")
        return

    print(f"  Found {len(branches_for_sample)} branch(es) for sample {target_sample_idx}")

    # ── load ALL prefix meta into a flat list, build prefix_id → meta slice ─
    prefix_meta_flat = []
    if has_prefix:
        for sid in range(num_p_shards):
            chunk = torch.load(raw_dir / "prefix_meta" / f"shard_{sid:04d}.pt",
                               weights_only=False)
            prefix_meta_flat.extend(chunk)
        prefix_lookup = {}
        for pid, (pstart, plen) in enumerate(pi):
            prefix_lookup[pid] = prefix_meta_flat[pstart: pstart + plen]

    # ── load ALL branch meta into a flat list ──────────────────────────────
    branch_meta_flat = []
    for sid in range(num_shards):
        chunk = torch.load(raw_dir / f"meta_shard_{sid:04d}.pt", weights_only=False)
        branch_meta_flat.extend(chunk)

    # ── helper: label symbol ───────────────────────────────────────────────
    def label_sym(is_correct):
        if is_correct is True:   return green("✓")
        if is_correct is False:  return red("✗")
        return yellow("?")

    # ── iterate branches ───────────────────────────────────────────────────
    for b_entry in branches_for_sample:
        b_start, b_len, sample_idx, prefix_id = b_entry

        branch_meta   = branch_meta_flat[b_start: b_start + b_len]
        prefix_meta_s = prefix_lookup[prefix_id] if has_prefix else []

        full_meta = prefix_meta_s + branch_meta

        # branch_label lives in branch_meta rows (may be absent for non-prm800k)
        branch_label = branch_meta[0].get("branch_label", "n/a") if branch_meta else "n/a"
        branch_rating = None
        # Try to infer rating from branch is_correct
        if branch_meta:
            ic = branch_meta[0].get("is_correct")
            branch_rating = ic

        print()
        print(bold(f"  ── Branch: {branch_label}  "
                   f"(prefix_id={prefix_id}, "
                   f"is_correct={branch_rating}) ──"))
        print(f"     prefix tokens: {len(prefix_meta_s)}  |  "
              f"branch tokens: {len(branch_meta)}  |  "
              f"total: {len(full_meta)}")

        # ── group by step_idx, print step summary ─────────────────────────
        steps = defaultdict(list)
        for m in full_meta:
            steps[m["step_idx"]].append(m)

        print(f"     Step breakdown:")
        for s_idx in sorted(steps):
            toks  = steps[s_idx]
            ic    = toks[0]["is_correct"]
            n_tok = len(toks)
            src   = "prefix" if toks[0] in prefix_meta_s else "branch"
            print(f"       step {s_idx:>3d}  {label_sym(ic)}  "
                  f"{n_tok:>4d} tokens  [{src}]")

        # ── detokenize if tokenizer is available ──────────────────────────
        if tokenizer is None:
            print(yellow("     (no tokenizer loaded — showing step counts only)"))
            continue

        # Build a pseudo token-id sequence by collecting token_pos order
        # We don't store raw token IDs in meta, so we reconstruct the text
        # by re-tokenizing per-step using the stored metadata text offsets.
        # Instead: we join per-step decoded fragments using tokenizer.decode
        # on the token_pos integers (they are positions, not ids — so we
        # print the metadata and decoded text side-by-side).
        # NOTE: token_pos is a LOCAL index within the reasoning region, not
        # a vocab token ID. So we can only show metadata here; for true
        # detokenization you'd need to re-run the tokenizer on the source text.
        # We print a warning to clarify this.
        print()
        print(yellow("     NOTE: token_pos in metadata is a position index, NOT a vocab id."))
        print(yellow("     To detokenize, re-run the tokenizer on the original dataset row."))
        print(yellow("     Showing per-step token position ranges instead:"))
        for s_idx in sorted(steps):
            if s_idx < 0:
                continue
            toks     = steps[s_idx]
            ic       = toks[0]["is_correct"]
            positions = [m["token_pos"] for m in toks]
            print(f"       step {s_idx:>3d}  {label_sym(ic)}  "
                  f"token_pos [{min(positions)} … {max(positions)}]  "
                  f"({len(positions)} tokens)")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5b – TRUE DETOKENIZATION from original JSONL
# ═════════════════════════════════════════════════════════════════════════════
def detokenize_from_source(raw_dir: Path, index: dict, target_sample_idx: int,
                           dataset_file: str, tokenizer):
    """Re-runs prepare_prm800k_branches on the original dataset row and
    pretty-prints each branch with colour-coded step labels."""

    print(bold(f"══ STEP 5b: True detokenization from source JSONL ══"))

    if not Path(dataset_file).exists():
        print(yellow(f"  Dataset file not found: {dataset_file}"))
        print(yellow("  Skipping step 5b. Pass --dataset_file to enable."))
        return

    import json

    with open(dataset_file, encoding="utf-8") as f:
        lines = f.readlines()

    if target_sample_idx >= len(lines):
        print(red(f"  sample_idx={target_sample_idx} out of range (dataset has {len(lines)} rows)"))
        return

    sample = json.loads(lines[target_sample_idx])
    print(f"  Problem: {sample['problem'][:120]}...")
    print(f"  Steps  : {len(sample['steps'])}")

    # Re-run the branch splitter from the main script logic
    # (inline minimal version so this file is self-contained)
    def split_branches(sample, tokenizer):
        steps = sample["steps"]
        branch_step_i = next(
            (si for si, s in enumerate(steps) if len(s.get("completions", [])) > 1),
            None
        )
        if branch_step_i is None:
            branch_step_i = len(steps) - 1  # fallback: last step

        prefix_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
        for si in range(branch_step_i):
            comp = steps[si]["completions"][0]
            prefix_text += comp.get("text", "") + "\n"

        branches = []
        for comp_j, comp in enumerate(steps[branch_step_i].get("completions", [])):
            rating = comp.get("rating", None)
            is_correct = True if rating in (0, 1) else (False if rating == -1 else None)
            branch_text = comp.get("text", "") + "\n"
            for si in range(branch_step_i + 1, len(steps)):
                comps = steps[si].get("completions", [])
                if comps:
                    branch_text += comps[0].get("text", "") + "\n"
            branches.append({
                "text":       branch_text,
                "rating":     rating,
                "is_correct": is_correct,
                "label":      f"step{branch_step_i}_comp{comp_j}",
            })
        return prefix_text, branches

    def label_sym(is_correct):
        if is_correct is True:   return green("✓")
        if is_correct is False:  return red("✗")
        return yellow("?")

    prefix_text, branches = split_branches(sample, tokenizer)

    prefix_ids = tokenizer(prefix_text)["input_ids"]
    print(f"\n  Shared prefix  ({len(prefix_ids)} tokens):")
    # print last 200 chars of prefix text to show the tail
    print(dim("    …" + prefix_text[-300:].replace("\n", "\n    ")))

    for b in branches:
        branch_ids = tokenizer(b["text"])["input_ids"]
        sym = label_sym(b["is_correct"])
        print(f"\n  {bold(b['label'])}  {sym}  rating={b['rating']}  ({len(branch_ids)} tokens)")
        # print full branch text, indented
        for line in b["text"].splitlines():
            print(f"    {line}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 – QUICK STATS from shard 0
# ═════════════════════════════════════════════════════════════════════════════
def quick_stats(raw_dir: Path, index: dict, layer: int):
    print(bold(f"══ STEP 6: Token class balance (shard 0, layer {layer}) ══"))

    meta_path = raw_dir / "meta_shard_0000.pt"
    if not meta_path.exists():
        print(yellow("  meta_shard_0000.pt not found, skipping."))
        return

    meta = torch.load(meta_path, weights_only=False)
    counts = defaultdict(int)
    for m in meta:
        counts[m["is_correct"]] += 1

    total = len(meta)
    print(f"  Total tokens in shard 0 : {total:,}")
    print(f"  Correct   (True)  : {counts[True]:>8,}  ({100*counts[True]/total:.1f}%)")
    print(f"  Incorrect (False) : {counts[False]:>8,}  ({100*counts[False]/total:.1f}%)")
    print(f"  Unlabelled (None) : {counts[None]:>8,}  ({100*counts[None]/total:.1f}%)")

    # Unique samples in shard 0
    sample_ids = {m["sample_idx"] for m in meta}
    step_keys  = {(m["sample_idx"], m["step_idx"]) for m in meta if m["step_idx"] >= 0}
    branch_labels = {m.get("branch_label") for m in meta if m.get("branch_label")}
    print(f"\n  Unique sample_idx in shard 0  : {len(sample_ids)}")
    print(f"  Unique (sample, step) pairs   : {len(step_keys)}")
    if branch_labels:
        print(f"  Unique branch_labels          : {len(branch_labels)}")
        print(f"  Branch labels (first 10)      : {sorted(branch_labels)[:10]}")

    # Unique problem_ids
    pids = {m.get("problem_id") for m in meta}
    print(f"  Unique problem_ids            : {len(pids)}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    if not raw_dir.exists():
        print(red(f"ERROR: raw_dir does not exist: {raw_dir}"))
        sys.exit(1)

    print(bold(f"\n{'═'*60}"))
    print(bold(f"  PRM800K raw_activations verifier"))
    print(bold(f"  dir   : {raw_dir}"))
    print(bold(f"  layer : {args.layer}"))
    print(bold(f"  sample: {args.sample_idx}"))
    print(bold(f"{'═'*60}"))

    # Load index once
    index_path = raw_dir / "index.pt"
    if not index_path.exists():
        print(red(f"ERROR: index.pt not found in {raw_dir}"))
        sys.exit(1)
    index = torch.load(index_path, weights_only=False)

    # ── run all checks ────────────────────────────────────────────────────
    check_structure(raw_dir, index)
    inspect_index(index)
    check_shard_shapes(raw_dir, index, args.layer)
    check_meta_alignment(raw_dir, index, args.layer)

    # Tokenizer (optional but recommended)
    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(bold("  Loading tokenizer…"))
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            print(green(f"  Tokenizer loaded: {args.model_name}\n"))
        except Exception as e:
            print(yellow(f"  Could not load tokenizer ({e}). Continuing without it.\n"))

    reconstruct_and_decode(raw_dir, index, args.sample_idx, tokenizer)

    # Step 5b: true detokenization if dataset file is provided
    dataset_file = getattr(args, "dataset_file",
                           "/home/ines/Reasoning-activations/reasoning_datasets/prm800k/"
                           "prm800k_phase2_test_cleaned_w_problem_ids.jsonl")
    if tokenizer is not None:
        detokenize_from_source(raw_dir, index, args.sample_idx, dataset_file, tokenizer)

    quick_stats(raw_dir, index, args.layer)

    print(bold("══ All checks complete ══\n"))


if __name__ == "__main__":
    main()