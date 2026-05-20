import json
import argparse
import sys

def export_dataset_errors(input_file, output_file, max_samples=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Filter the dataset
    filtered_samples = [
        sample for sample in dataset
        if sample.get("label") != -1 and sample.get("final_answer_correct") is False
    ]

    print(f"Found {len(filtered_samples)} samples matching the criteria.")
    print(f"Writing to '{output_file}'...")

    # Limit samples if requested
    if max_samples:
        filtered_samples = filtered_samples[:max_samples]

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(f"# ProcessBench Error Analysis\n")
            out_f.write(f"Total matching samples found: {len(filtered_samples)}\n")
            out_f.write("=" * 80 + "\n\n")

            for i, sample in enumerate(filtered_samples):
                out_f.write(f"## --- Sample {i + 1} ---\n\n")
                out_f.write(f"**Problem:**\n")
                out_f.write(f"{sample['problem']}\n\n")
                
                out_f.write(f"**Reasoning Steps:**\n")
                error_label = sample['label']
                
                for step_idx, step_text in enumerate(sample['steps']):
                    # Highlight the step if its index matches the error label
                    if step_idx == error_label:
                        out_f.write(f"\n>> **[FIRST ERRONEOUS STEP - Step {step_idx}]:** {step_text} <<\n\n")
                    else:
                        out_f.write(f"   Step {step_idx}: {step_text}\n")
                        
                out_f.write("\n" + "=" * 80 + "\n\n")
                
        print("Export complete!")
        
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export erroneous reasoning steps to a text/markdown file.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="/home/ines/Reasoning-activations/reasoning_datasets/ProcessBench/dataset.jsonl", 
        help="Path to the input dataset.jsonl file (default: dataset.jsonl)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="erroneous_steps_output.md", 
        help="Path for the output text/markdown file (default: erroneous_steps_output.md)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Maximum number of samples to export"
    )
    
    args = parser.parse_args()
    export_dataset_errors(args.input, args.output, args.limit)