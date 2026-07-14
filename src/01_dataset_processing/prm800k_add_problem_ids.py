import json
import sys

def add_problem_ids(input_path: str, output_path: str):
    # Load all records
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Map problem text -> assigned problem_id (ordinal, starting at 0)
    problem_to_id: dict[str, int] = {}
    next_id = 0

    for record in records:
        problem_text = record["problem"]
        if problem_text not in problem_to_id:
            problem_to_id[problem_text] = next_id
            next_id += 1
        record["problem_id"] = problem_to_id[problem_text]

    # Write output
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Processed {len(records)} rows → {next_id} unique problems.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_problem_ids.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    add_problem_ids(sys.argv[1], sys.argv[2])