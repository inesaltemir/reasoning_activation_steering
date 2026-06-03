"""
Parse PRM800K phase2_test.jsonl
================================

Rads from local path) the PRM800K phase2_test.jsonl dataset and
produces a cleaned JSONL file with the following filters applied:

  1. Remove any sample whose finish_reason is "bad_problem" or "give_up".
  2. Within each remaining sample, remove any completion where flagged is True.
  3. If a step has zero completions remaining after filtering, remove that step.
  4. If a sample has zero steps remaining after filtering, remove that sample.

Output schema (one JSON object per line):
  {
    "problem":                <str>,
    "ground_truth_solution":  <str>,
    "ground_truth_answer":    <str>,
    "steps": [
      {
        "completions": [
          {"text": <str>, "rating": <int>, "flagged": <bool|null>}
        ],
        "chosen_completion": <int|null>,
        "human_completion":  <str|null>
      }
    ],
    "finish_reason": <str>
  }

Optionally, if --max_samples is provided, a random subset of that size is
sampled from the cleaned dataset.

Usage:
  python parse_prm800k.py --output_file prm800k_cleaned.jsonl
  python parse_prm800k.py --output_file prm800k_cleaned.jsonl --max_samples 500
  python parse_prm800k.py --input_file /path/to/phase2_test.jsonl --output_file out.jsonl
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean and filter PRM800K phase2_test.jsonl."
    )
    p.add_argument(
        "--input_file",
        type=str,
        default="/home/ines/Reasoning-activations/reasoning_datasets/prm800k/phase2_test.jsonl",
        help="Path to a local phase2_test.jsonl.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default="/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned.jsonl",
        help="Path to the output .jsonl file.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, randomly sample this many rows from the cleaned dataset.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (only used with --max_samples).",
    )
    return p.parse_args()


def filter_completions(completions: list[dict]) -> list[dict]:
    """Remove completions where flagged is True (keep flagged=False or None)."""
    return [c for c in completions if c.get("flagged") is not True]


def clean_sample(raw: dict) -> dict | None:
    """
    Apply all cleaning rules to a single sample.

    Returns None if the sample should be discarded entirely.
    """
    # --- 1. Discard by finish_reason ---
    finish_reason = raw.get("label", {}).get("finish_reason")
    if finish_reason in ("bad_problem", "give_up"):
        return None

    # --- 2. Build cleaned steps ---
    raw_steps = raw.get("label", {}).get("steps", [])
    cleaned_steps = []

    for step in raw_steps:
        raw_completions = step.get("completions", [])
        kept = filter_completions(raw_completions)

        if not kept:
            # All completions were flagged — drop the entire step
            continue

        # chosen_completion is always 0 or None in PRM800K.
        # If 0, the chosen one is raw_completions[0]; since filtering
        # preserves order, it remains kept[0] if it wasn't flagged.
        original_chosen = step.get("chosen_completion")
        new_chosen = (
            0 if original_chosen == 0
            and raw_completions[0].get("flagged") is not True
            else None
        )

        cleaned_steps.append({
            "completions": [
                {
                    "text": c["text"],
                    "rating": c["rating"],
                    "flagged": c.get("flagged"),
                }
                for c in kept
            ],
            "chosen_completion": new_chosen,
            "human_completion": step.get("human_completion"),
        })

    if not cleaned_steps:
        return None

    # --- 3. Assemble output record ---
    question = raw.get("question", {})
    return {
        "problem": question.get("problem", ""),
        "ground_truth_solution": question.get("ground_truth_solution", ""),
        "ground_truth_answer": question.get("ground_truth_answer", ""),
        "steps": cleaned_steps,
        "finish_reason": finish_reason,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    # --- Resolve input file ---
    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # --- Read & clean ---
    cleaned = []
    stats = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            stats["total"] += 1

            result = clean_sample(raw)
            if result is None:
                reason = raw.get("label", {}).get("finish_reason", "unknown")
                stats[f"dropped_{reason}"] += 1
                continue

            stats["kept"] += 1
            cleaned.append(result)

    print(f"\n--- Filtering summary ---")
    print(f"  Total samples read:   {stats['total']}")
    print(f"  Kept after cleaning:  {stats['kept']}")
    for key in sorted(k for k in stats if k.startswith("dropped_")):
        print(f"  {key:>25s}: {stats[key]}")

    # --- Optional subsampling ---
    if args.max_samples is not None and args.max_samples < len(cleaned):
        random.shuffle(cleaned)
        cleaned = cleaned[: args.max_samples]
        print(f"  Sampled {args.max_samples} rows from {stats['kept']} cleaned rows")

    # --- Write output ---
    with open(args.output_file, "w", encoding="utf-8") as out:
        for rec in cleaned:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Final stats ---
    total_steps = sum(len(r["steps"]) for r in cleaned)
    total_completions = sum(
        len(c) for r in cleaned for c in (s["completions"] for s in r["steps"])
    )
    finish_reasons = Counter(r["finish_reason"] for r in cleaned)

    print(f"\n--- Output summary ---")
    print(f"  Output file:        {args.output_file}")
    print(f"  Samples written:    {len(cleaned)}")
    print(f"  Total steps:        {total_steps}")
    print(f"  Total completions:  {total_completions}")
    print(f"  Finish reasons:")
    for reason, count in finish_reasons.most_common():
        print(f"    {reason:>20s}: {count}")

    # Rating distribution
    rating_counts = Counter()
    for r in cleaned:
        for s in r["steps"]:
            for c in s["completions"]:
                rating_counts[c["rating"]] += 1
    print(f"  Rating distribution:")
    for rating in sorted(rating_counts):
        print(f"    {rating:>3d}: {rating_counts[rating]}")


if __name__ == "__main__":
    main()