import json

def expand_row(row):
    """Expand a row into multiple rows if any step has multiple completions (chosen_completion is None)."""
    steps = row["steps"]
    
    # Find the first step with multiple completions and no chosen_completion
    split_step_idx = None
    for i, step in enumerate(steps):
        if step["chosen_completion"] is None and len(step["completions"]) > 1:
            split_step_idx = i
            break
    
    if split_step_idx is None:
        # No split needed, return as-is
        return [row]
    
    results = []
    split_step = steps[split_step_idx]
    
    for completion in split_step["completions"]:
        new_row = {
            "problem": row["problem"],
            "ground_truth_solution": row["ground_truth_solution"],
            "ground_truth_answer": row["ground_truth_answer"],
            "steps": []
        }
        
        # Copy steps up to (not including) the split step unchanged
        for step in steps[:split_step_idx]:
            new_row["steps"].append(step)
        
        # Add the split step with only this single completion
        new_row["steps"].append({
            "completions": [completion],
            "chosen_completion": None,
            "human_completion": split_step["human_completion"]
        })
        
        results.append(new_row)
    
    return results

# Read input
rows = []
with open("/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Expand all rows
output_rows = []
for row in rows:
    output_rows.extend(expand_row(row))

# Write output
with open("/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned_multiple_traj.jsonl", "w") as f:
    for row in output_rows:
        f.write(json.dumps(row) + "\n")

print(f"Input rows: {len(rows)}")
print(f"Output rows: {len(output_rows)}")