import json


def expand_row(row, problem_id):
    """Expand a row into multiple rows if any step has multiple completions (chosen_completion is None).
    
    Each output row gets a `problem_id` (the 0-based index of the original input row) and a
    `variant` (0-based ordinal among the expanded copies; 0 for rows that needed no splitting).
    """
    steps = row["steps"]

    # Find the first step with multiple completions and no chosen_completion
    split_step_idx = None
    for i, step in enumerate(steps):
        if step["chosen_completion"] is None and len(step["completions"]) > 1:
            split_step_idx = i
            break

    # Base fields shared by every output row derived from this input row
    base = {
        "problem_id": problem_id,
        "problem": row["problem"],
        "ground_truth_solution": row["ground_truth_solution"],
        "ground_truth_answer": row["ground_truth_answer"],
    }

    if split_step_idx is None:
        # No split needed — single output row, variant = 0
        return [{**base, "variant": 0, "steps": steps}]

    results = []
    split_step = steps[split_step_idx]

    for variant, completion in enumerate(split_step["completions"]):
        new_row = {
            **base,
            "variant": variant,
            "steps": (
                # All steps before the split, unchanged
                list(steps[:split_step_idx])
                + [
                    # The split step with only this single completion
                    {
                        "completions": [completion],
                        "chosen_completion": None,
                        "human_completion": split_step["human_completion"],
                    }
                ]
            ),
        }
        results.append(new_row)

    return results


# Read input
rows = []
with open("/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Expand all rows, passing the 0-based row index as problem_id
output_rows = []
for problem_id, row in enumerate(rows):
    output_rows.extend(expand_row(row, problem_id))

# Write output
with open("/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned_deduplicated_traj.jsonl", "w") as f:
    for row in output_rows:
        f.write(json.dumps(row) + "\n")

print(f"Input rows:  {len(rows)}")
print(f"Output rows: {len(output_rows)}")