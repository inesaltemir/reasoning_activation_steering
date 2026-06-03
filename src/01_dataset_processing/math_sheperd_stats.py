"""
Math-Shepherd Dataset Statistics
================================
Reads a .jsonl file (output of parse_math_shepherd.py) and prints
summary statistics + histograms.

Usage:
  python math_shepherd_stats.py --input math_shepherd_dataset.jsonl
  python math_shepherd_stats.py --input math_shepherd_dataset.jsonl --max_position 15
"""

import argparse
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/math-sheperd/math_shepherd_dataset_3000samples.jsonl")
    parser.add_argument("--max_position", type=int, default=30,
                        help="Max step position to show in per-position plot (default: 20)")
    parser.add_argument("--output", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/math-sheperd/stats_3000samples.png",
                        help="Save figure to this path (e.g. stats.png). Shows interactively if omitted.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} samples from {args.input}\n")

    # ── Collect stats ──
    num_steps_list = []
    correct_per_sample = []
    incorrect_per_sample = []
    correct_at_pos = Counter()    # position -> count of correct
    incorrect_at_pos = Counter()  # position -> count of incorrect
    task_counts = Counter()
    label_counts = Counter()      # all-correct vs has-error

    for rec in records:
        n = rec["num_steps"]
        sl = rec["step_labels"]
        num_steps_list.append(n)
        correct_per_sample.append(sum(sl))
        incorrect_per_sample.append(sum(1 for s in sl if not s))
        task_counts[rec.get("task", "unknown")] += 1
        label_counts["all_correct" if rec["label"] == -1 else "has_error"] += 1

        for pos, is_correct in enumerate(sl):
            if is_correct:
                correct_at_pos[pos] += 1
            else:
                incorrect_at_pos[pos] += 1

    num_steps_arr = np.array(num_steps_list)
    correct_arr = np.array(correct_per_sample)
    incorrect_arr = np.array(incorrect_per_sample)

    # ── Print summary ──
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Samples:            {len(records)}")
    for task, cnt in sorted(task_counts.items()):
        print(f"    {task:>10}: {cnt}")
    print(f"  All-correct:        {label_counts['all_correct']}")
    print(f"  Has-error:          {label_counts['has_error']}")
    print(f"  Steps per sample:   mean={num_steps_arr.mean():.1f}  "
          f"median={np.median(num_steps_arr):.0f}  "
          f"min={num_steps_arr.min()}  max={num_steps_arr.max()}")
    total_steps = correct_arr.sum() + incorrect_arr.sum()
    print(f"  Total steps:        {total_steps}")
    print(f"    correct:          {correct_arr.sum()}  ({100*correct_arr.sum()/total_steps:.1f}%)")
    print(f"    incorrect:        {incorrect_arr.sum()}  ({100*incorrect_arr.sum()/total_steps:.1f}%)")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Math-Shepherd Dataset Stats  (n={len(records)})", fontsize=14, fontweight="bold")

    # (a) Histogram: number of steps per sample
    ax = axes[0, 0]
    max_bin = min(int(num_steps_arr.max()), 30)
    ax.hist(num_steps_arr, bins=range(1, max_bin + 2), edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Number of samples")
    ax.set_title("Steps per sample")
    ax.axvline(num_steps_arr.mean(), color="red", ls="--", label=f"mean={num_steps_arr.mean():.1f}")
    ax.legend()

    # (b) Histogram: correct vs incorrect steps per sample
    ax = axes[0, 1]
    bins = range(0, max(int(correct_arr.max()), int(incorrect_arr.max())) + 2)
    ax.hist(correct_arr, bins=bins, alpha=0.6, label="correct", color="#55A868", edgecolor="black")
    ax.hist(incorrect_arr, bins=bins, alpha=0.6, label="incorrect", color="#C44E52", edgecolor="black")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Number of samples")
    ax.set_title("Correct / incorrect steps per sample")
    ax.legend()

    # (c) Stacked bar: correct vs incorrect at each step position
    ax = axes[1, 0]
    max_pos = args.max_position
    positions = list(range(max_pos))
    corr_vals = [correct_at_pos.get(p, 0) for p in positions]
    incorr_vals = [incorrect_at_pos.get(p, 0) for p in positions]
    x = np.arange(max_pos)
    ax.bar(x, corr_vals, label="correct", color="#55A868", edgecolor="black", width=0.8)
    ax.bar(x, incorr_vals, bottom=corr_vals, label="incorrect", color="#C44E52", edgecolor="black", width=0.8)
    ax.set_xlabel("Step position (0-indexed)")
    ax.set_ylabel("Count")
    ax.set_title("Correct / incorrect steps by position")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p + 1) for p in positions], fontsize=8)
    ax.legend()

    # (d) Fraction incorrect at each position
    ax = axes[1, 1]
    frac_incorrect = []
    for p in positions:
        total = correct_at_pos.get(p, 0) + incorrect_at_pos.get(p, 0)
        frac_incorrect.append(incorrect_at_pos.get(p, 0) / total if total > 0 else 0)
    ax.bar(x, frac_incorrect, color="#DD8452", edgecolor="black", width=0.8)
    ax.set_xlabel("Step position (0-indexed)")
    ax.set_ylabel("Fraction incorrect")
    ax.set_title("Error rate by step position")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p + 1) for p in positions], fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()