"""
Parse Math-Shepherd Dataset into ProcessBench-Compatible JSONL
===============================================================

Loads the Math-Shepherd dataset from HuggingFace:
    https://huggingface.co/datasets/peiyi9979/Math-Shepherd

Format details (from the paper & HF page):
  - "input":  contains BOTH the problem text AND the step-by-step solution,
              with "ки" as a step-boundary token (instead of +/-).
  - "label":  same text but with +/- replacing "ки" to mark correctness.
  - "task":   "GSM8K" or "MATH".

The problem definition is the text *before* "Step 1:" in the input field.
Steps are delimited by "Step N:" markers; each ends with + or - in the label.

Output JSONL schema (one JSON object per line):
  {
    "problem_id":   "mathshepherd_<idx>",
    "problem":      "<problem text>",
    "steps":        ["<step 1 text>", "<step 2 text>", ...],
    "step_labels":  [true, false, ...],       # per-step correctness
    "label":        <int>,                     # first error index, -1 if all correct
    "num_steps":    <int>,
    "task":         "GSM8K" | "MATH",
    "source":       "math-shepherd"
  }

This is directly compatible with prepare_prompt_and_labels_processbench()
in run_fw_pass_with_step_averaging_storage.py, which builds the prompt:

    Problem:
    {problem}

    Reasoning:
    Step 1: {steps[0]}
    Step 2: {steps[1]}
    ...

and derives per-step correctness from the `label` field.

Usage:
  pip install datasets
  python parse_math_shepherd.py --output_file math_shepherd_dataset.jsonl
  python parse_math_shepherd.py --output_file math_shepherd_dataset.jsonl --max_samples 5000 --balanced
"""

import argparse
import json
import re
import sys
import random
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse Math-Shepherd into ProcessBench-compatible JSONL."
    )
    p.add_argument("--output_file", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/math-sheperd/math_shepherd_dataset_3000samples.jsonl",
                   help="Path to the output .jsonl file.")
    p.add_argument("--max_samples", type=int, default=3000,
                   help="Maximum number of samples to write.")
    p.add_argument("--balanced", action="store_true", default=False,
                   help="Enforce equal counts of GSM8K and MATH samples. "
                        "When --max_samples is set, each task gets max_samples // 2.")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to load (default: train).")
    p.add_argument("--streaming", action="store_true", default=False,
                   help="Use streaming mode (saves RAM for the full dataset).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffling before balanced sampling.")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Print per-sample parsing diagnostics.")
    return p.parse_args()


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────


STEP_PATTERN = re.compile(r"Step\s+(\d+)\s*:")


def extract_problem(input_text: str) -> str:
    """Extract the problem definition = everything before 'Step 1:' in the input field."""
    match = STEP_PATTERN.search(input_text)
    if match is None:
        return input_text.strip()
    return input_text[:match.start()].strip()


def _find_label_backward(text: str, end_pos: int):
    """
    Scan backward from end_pos in *text* (exclusive) to find the +/- label,
    skipping trailing whitespace.

    Returns
    -------
    label : bool | None
        True (+), False (-), or None if not found.
    label_char_pos : int
        Index of the +/- character in *text* (-1 if not found).
    """
    pos = end_pos - 1
    while pos >= 0 and text[pos] in (" ", "\t", "\n", "\r"):
        pos -= 1
    if pos >= 0 and text[pos] == "+":
        return True, pos
    elif pos >= 0 and text[pos] == "-":
        return False, pos
    return None, -1


def parse_steps_from_label(label_text: str):
    """
    Parse the 'label' field of a Math-Shepherd sample.

    The label field mirrors the input but replaces the 'ки' token with
    '+' (correct) or '-' (incorrect) at each step boundary:
      - For step i (not the last): the +/- sits right before "Step {i+1}:".
      - For the last step: the +/- is the last non-whitespace char of the string.

    Returns
    -------
    steps : list[str]
        Step text content (without "Step N:" prefix or +/- suffix).
    step_labels : list[bool]
        True = correct (+), False = incorrect (-).
    """
    matches = list(STEP_PATTERN.finditer(label_text))
    if not matches:
        return [], []

    steps = []
    step_labels = []

    for i, match in enumerate(matches):
        content_start = match.end()

        if i + 1 < len(matches):
            # Non-last step: label is right before the next "Step N:" marker
            boundary = matches[i + 1].start()
        else:
            # Last step: label is at the end of the full string
            boundary = len(label_text)

        label, label_pos = _find_label_backward(label_text, boundary)

        if label is not None and label_pos >= content_start:
            step_text = label_text[content_start:label_pos].strip()
        else:
            # Fallback: take everything, mark as None
            step_text = label_text[content_start:boundary].strip()

        steps.append(step_text)
        step_labels.append(label)

    return steps, step_labels


def first_error_index(step_labels: list) -> int:
    """
    Index of the first incorrect step, or -1 if all correct.
    Matches ProcessBench's 'label' field convention used by
    prepare_prompt_and_labels_processbench().
    """
    for i, is_correct in enumerate(step_labels):
        if is_correct is False:
            return i
    return -1


def parse_one_sample(sample: dict, idx: int, verbose: bool = False):
    """
    Parse a single Math-Shepherd sample dict into the output record.
    Returns None if the sample cannot be parsed.
    """
    input_text = sample["input"].strip()
    label_text = sample["label"].strip()
    task = sample.get("task", "unknown")

    # 1. Problem = text before first "Step N:" in the input field
    problem = extract_problem(input_text)
    if not problem:
        return None

    # 2. Steps + labels from the label field (which has +/- markers)
    steps, step_labels = parse_steps_from_label(label_text)

    if len(steps) == 0:
        if verbose:
            print(f"  [SKIP] Sample {idx}: no steps parsed.")
        return None
    if None in step_labels:
        if verbose:
            print(f"  [SKIP] Sample {idx}: step with missing +/- label.")
        return None

    # 3. ProcessBench-compatible first-error label
    label = first_error_index(step_labels)

    record = {
        "problem_id": f"mathshepherd_{idx}",
        "problem": problem,
        "steps": steps,
        "step_labels": step_labels,
        "label": label,
        "num_steps": len(steps),
        "task": task,
        "source": "math-shepherd",
    }

    if verbose and idx < 5:
        print(f"\n--- Sample {idx} (task={task}) ---")
        print(f"  Problem:    {problem[:120]}...")
        print(f"  # Steps:    {len(steps)}")
        for si, (st, sl) in enumerate(zip(steps, step_labels)):
            tag = "+" if sl else "-"
            print(f"    Step {si}: [{tag}] {st[:90]}{'...' if len(st)>90 else ''}")
        print(f"  First err:  {label}")

    return record


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the datasets library:  pip install datasets")
        sys.exit(1)

    print(f"Loading Math-Shepherd (split={args.split}, streaming={args.streaming})...")
    ds = load_dataset(
        "peiyi9979/Math-Shepherd",
        split=args.split,
        streaming=args.streaming,
    )

    # ── Parse all (or stream up to a generous cap) ──
    # When balanced, we need to collect by task first, then subsample.
    records_by_task = {"GSM8K": [], "MATH": [], "unknown": []}
    parse_stats = Counter()
    total_seen = 0

    # If not balanced, we can write eagerly when not sampling
    # If balanced or max_samples, collect first then write
    need_collect = args.balanced or args.max_samples is not None

    out_f = None
    if not need_collect:
        out_f = open(args.output_file, "w", encoding="utf-8")

    for idx, sample in enumerate(ds):
        total_seen += 1

        record = parse_one_sample(sample, idx, verbose=args.verbose)
        if record is None:
            parse_stats["skipped"] += 1
            continue

        parse_stats["parsed"] += 1
        task = record["task"]

        if need_collect:
            records_by_task.setdefault(task, []).append(record)
        else:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress
        if total_seen % 50_000 == 0:
            print(f"  ... processed {total_seen} raw samples "
                  f"({parse_stats['parsed']} parsed, {parse_stats['skipped']} skipped)")

    if out_f is not None:
        out_f.close()

    print(f"\nParsing done: {parse_stats['parsed']} valid / "
          f"{parse_stats['skipped']} skipped / {total_seen} total seen")
    for task, recs in sorted(records_by_task.items()):
        if recs:
            print(f"  {task}: {len(recs)} samples")

    # ── Balanced / max_samples selection ──
    if need_collect:
        gsm = records_by_task.get("GSM8K", [])
        math = records_by_task.get("MATH", [])
        other = records_by_task.get("unknown", [])

        random.shuffle(gsm)
        random.shuffle(math)
        random.shuffle(other)

        if args.balanced:
            if args.max_samples is not None:
                per_task = args.max_samples // 2
            else:
                per_task = min(len(gsm), len(math))
            selected = gsm[:per_task] + math[:per_task]
            print(f"\nBalanced selection: {per_task} GSM8K + {per_task} MATH "
                  f"= {len(selected)} total")
        else:
            # Not balanced, but max_samples is set
            all_records = gsm + math + other
            random.shuffle(all_records)
            selected = all_records[:args.max_samples]
            print(f"\nSelected {len(selected)} samples (max_samples={args.max_samples})")

        # Shuffle final order and re-index problem_ids
        random.shuffle(selected)
        for i, rec in enumerate(selected):
            rec["problem_id"] = f"mathshepherd_{i}"

        # Write
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            for rec in selected:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Final statistics ──
    # Re-read for summary stats
    stats = Counter()
    with open(args.output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["samples"] += 1
            stats[f"task_{rec['task']}"] += 1
            stats["total_steps"] += rec["num_steps"]
            stats["correct_steps"] += sum(rec["step_labels"])
            stats["incorrect_steps"] += sum(1 for s in rec["step_labels"] if not s)
            if rec["label"] == -1:
                stats["all_correct_samples"] += 1
            else:
                stats["has_error_samples"] += 1

    print(f"\n{'='*50}")
    print(f"Output: {args.output_file}")
    print(f"  Total samples:         {stats['samples']}")
    for key in sorted(k for k in stats if k.startswith("task_")):
        print(f"    {key.replace('task_',''):>10}: {stats[key]}")
    print(f"  All-correct (label=-1): {stats['all_correct_samples']}")
    print(f"  Has-error  (label>=0): {stats['has_error_samples']}")
    print(f"  Total steps:           {stats['total_steps']}")
    print(f"    correct:             {stats['correct_steps']}")
    print(f"    incorrect:           {stats['incorrect_steps']}")


if __name__ == "__main__":
    main()