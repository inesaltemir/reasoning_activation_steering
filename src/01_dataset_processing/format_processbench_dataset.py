import json
import logging
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
# The official Hugging Face dataset name for ProcessBench
HF_DATASET_NAME = "Qwen/ProcessBench" 
SPLIT = "omnimath"  # ProcessBench provides: gsm8k (400), math (1k), olympiadbench, omnimath
OUTPUT_FILE = "omnimath_processbench_formatted.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def format_dataset():
    logging.info(f"Downloading dataset '{HF_DATASET_NAME}' (split: {SPLIT})...")
    try:
        dataset = load_dataset(HF_DATASET_NAME, split=SPLIT)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.info("If the dataset is gated, make sure you are authenticated via `huggingface-cli login`.")
        return

    formatted_data = []
    
    logging.info("Formatting dataset...")
    for row in tqdm(dataset, desc="Processing rows"):
        # ProcessBench schema mapping. 
        try:
            formatted_sample = {
                "problem": row.get("problem"),
                "steps": row.get("steps"),
                # If the label represents the index of the first incorrect step:
                # Some versions of ProcessBench use -1 to indicate all steps are correct.
                "label": int(row.get("label")), 
                "final_answer_correct": bool(row.get("final_answer_correct"))
            }
            
            # Basic validation to ensure we don't save malformed data
            if formatted_sample["problem"] and isinstance(formatted_sample["steps"], list):
                formatted_data.append(formatted_sample)
                
        except Exception as e:
            logging.warning(f"Skipping a malformed row: {e}")
            continue

    logging.info(f"Saving {len(formatted_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in formatted_data:
            f.write(json.dumps(sample) + "\n")
            
    logging.info("Dataset formatting complete!")

if __name__ == "__main__":
    format_dataset()