"""
Visualise Math-Shepherd samples as a readable markdown file,
highlighting each step in green (correct) or red (incorrect).

Usage:
  python visualise_math_shepherd.py --input math_shepherd_dataset.jsonl
  python visualise_math_shepherd.py --input math_shepherd_dataset.jsonl --limit 50 --errors_only
"""

import json
import argparse
import sys


def export_samples(input_file, output_file, max_samples=None, errors_only=False):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)

    if errors_only:
        dataset = [s for s in dataset if s.get("label", -1) != -1]

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"Writing {len(dataset)} samples → {output_file}")

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"# Math-Shepherd Sample Viewer\n")
        out_f.write(f"Samples shown: {len(dataset)}\n")
        out_f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(dataset):
            task = sample.get("task", "")
            label = sample.get("label", -1)
            tag = "ALL CORRECT" if label == -1 else f"FIRST ERROR @ Step {label + 1}"

            out_f.write(f"## --- Sample {i + 1}  [{task}]  ({tag}) ---\n\n")
            out_f.write(f"**Problem:**\n{sample['problem']}\n\n")
            out_f.write(f"**Steps:**\n")

            step_labels = sample.get("step_labels", [])
            for idx, step_text in enumerate(sample["steps"]):
                is_correct = step_labels[idx] if idx < len(step_labels) else None

                if is_correct is True:
                    out_f.write(f"   ✅ Step {idx + 1}: {step_text}\n")
                elif is_correct is False:
                    out_f.write(f"   ❌ Step {idx + 1}: {step_text}\n")
                else:
                    out_f.write(f"   ❓ Step {idx + 1}: {step_text}\n")

            out_f.write("\n" + "=" * 80 + "\n\n")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise Math-Shepherd samples as markdown.")
    parser.add_argument("--input", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/math_shepherd/math_shepherd_dataset_test.jsonl")
    parser.add_argument("--output", type=str, default="/home/ines/Reasoning-activations/math_shepherd_viewer3.md", help="Output markdown file")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to show")
    parser.add_argument("--errors_only", action="store_true", help="Only show samples with at least one error")
    args = parser.parse_args()
    export_samples(args.input, args.output, args.limit, args.errors_only)