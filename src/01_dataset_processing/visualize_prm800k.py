import json
import argparse
import sys


RATING_LABELS = {
    1:  "✅ correct",
    0:  "⚠️  neutral",
    -1: "❌ erroneous",
}


def render_step(out_f, step_idx: int, step: dict, is_last: bool) -> None:
    """Write a single step block to the output file."""
    completions = step.get("completions", [])
    chosen = step.get("chosen_completion")

    for comp_idx, comp in enumerate(completions):
        text   = comp.get("text", "").strip()
        rating = comp.get("rating")
        label  = RATING_LABELS.get(rating, f"rating={rating}")
        is_chosen = (chosen is not None and comp_idx == chosen)

        chosen_marker = " *(chosen)*" if is_chosen else ""

        if rating == -1:
            # Erroneous step — call it out visually
            out_f.write(
                f"\n> **[ERRONEOUS STEP {step_idx}]{chosen_marker}** {label}  \n"
                f"> {text}\n\n"
            )
        elif rating == 1:
            out_f.write(f"**Step {step_idx}**{chosen_marker} {label}  \n{text}\n\n")
        else:
            out_f.write(f"**Step {step_idx}**{chosen_marker} {label}  \n{text}\n\n")


def has_error(sample: dict) -> bool:
    """Return True if at least one completion in any step has rating == -1."""
    for step in sample.get("steps", []):
        for comp in step.get("completions", []):
            if comp.get("rating") == -1:
                return True
    return False


def export_prm800k(input_file: str, output_file: str,
                   errors_only: bool = False, limit: int = None) -> None:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)

    if errors_only:
        samples = [s for s in samples if has_error(s)]
        print(f"Filtered to {len(samples)} samples containing at least one erroneous step.")
    else:
        print(f"Loaded {len(samples)} samples.")

    if limit:
        samples = samples[:limit]

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write("# PRM800K Dataset Viewer\n\n")
            out_f.write(f"**Source:** `{input_file}`  \n")
            out_f.write(f"**Samples shown:** {len(samples)}")
            if errors_only:
                out_f.write("  *(errors-only filter active)*")
            out_f.write("\n\n---\n\n")

            for i, sample in enumerate(samples):
                problem        = sample.get("problem", "*(no problem text)*")
                gt_answer      = sample.get("ground_truth_answer", "—")
                gt_solution    = sample.get("ground_truth_solution", "")
                finish_reason  = sample.get("finish_reason", "")
                steps          = sample.get("steps", [])

                # ── Header ──────────────────────────────────────────────────
                out_f.write(f"## Sample {i + 1}\n\n")
                out_f.write(f"**Problem:** {problem}\n\n")
                out_f.write(f"**Ground-truth answer:** `{gt_answer}`\n\n")
                if gt_solution:
                    out_f.write(f"<details><summary>Ground-truth solution</summary>\n\n{gt_solution}\n\n</details>\n\n")
                if finish_reason:
                    out_f.write(f"*Finish reason: {finish_reason}*\n\n")

                # ── Steps ────────────────────────────────────────────────────
                out_f.write("### Reasoning steps\n\n")
                if not steps:
                    out_f.write("*(no steps)*\n\n")
                else:
                    for step_idx, step in enumerate(steps):
                        render_step(out_f, step_idx, step, is_last=(step_idx == len(steps) - 1))

                out_f.write("---\n\n")

    except IOError as e:
        print(f"Error writing to '{output_file}': {e}")
        sys.exit(1)

    print(f"Done → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise a PRM800K-format JSONL file as clean Markdown, "
                    "with erroneous steps clearly marked."
    )
    parser.add_argument("--input",  "-i", default="/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned_multiple_traj.jsonl",
                        help="Path to the input .jsonl file.")
    parser.add_argument("--output", "-o", default="prm800k_viz.md",
                        help="Path for the output Markdown file (default: prm800k_viz.md).")
    parser.add_argument("--errors-only", action="store_true",
                        help="Only include samples that contain at least one erroneous step.")
    parser.add_argument("--limit", "-n", type=int, default=10,
                        help="Maximum number of samples to render.")

    args = parser.parse_args()
    export_prm800k(args.input, args.output, args.errors_only, args.limit)