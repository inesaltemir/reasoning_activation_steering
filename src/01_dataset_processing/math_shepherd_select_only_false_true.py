import json

input_file = "/home/ines/Reasoning-activations/reasoning_datasets/math_shepherd/math_shepherd_dataset_3000samples.jsonl"
output_file = "/home/ines/Reasoning-activations/reasoning_datasets/math_shepherd/math_shepherd_data_false_before_true.jsonl"

def has_false_before_true(step_labels):
    """
    Returns True if there exists at least one False
    occurring before a later True in the list.
    """
    seen_false = False

    for label in step_labels:
        if label is False:
            seen_false = True
        elif label is True and seen_false:
            return True

    return False


with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        row = json.loads(line)

        step_labels = row.get("step_labels", [])

        if has_false_before_true(step_labels):
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Filtered dataset saved to: {output_file}")