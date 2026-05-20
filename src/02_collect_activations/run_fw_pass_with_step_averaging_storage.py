"""
Script to run forward passes and extract layer-wise activations for the model in two distinct modes: 'reasoning' and 'baseline'. 
- In 'reasoning' mode, it aligns tokens with logical steps from the ProcessBench dataset to compute mean activation vectors for correct and incorrect reasoning tokens/steps. 
- In 'baseline' mode, it extracts base activations from the FineWeb dataset and computes PCA components to capture baseline language variance.

Type of output:
a `.pt` reasoning-vectors file, where for each layer we store:
  - `mean_correct_token`: global average of every correct reasoning token `[d_model]`
  - `mean_incorrect_token`: global average of every incorrect reasoning token `[d_model]`
  - `reasoning_direction_token`: `mean_correct_token - mean_incorrect_token` `[d_model]`
  - `per_sample_means`: per-sample average activation `[N_samples, d_model]`
  - `mean_correct_samples`: mean activation of perfectly-correct samples `[d_model]`
  - `mean_incorrect_samples`: mean activation of flawed samples `[d_model]`
  - `reasoning_direction_sample`: `mean_correct_samples - mean_incorrect_samples` `[d_model]`
  - `mean_correct_step`:          mean_correct_step,
  - `mean_incorrect_step`:        mean_incorrect_step,
  - `reasoning_direction_step`:   mean_correct_step - mean_incorrect_step,

Note: we DO save the raw activation value for each token, but only global means for (in)correct tokens, steps and samples. 

Structure:
- Arguments: can specifiy '--type reasoning' and '--type baseline'.
- Helper function for reasoning dataset ProcessBench: `prepare_prompt_and_labels_processbench` aligns character offsets to specific reasoning steps to label tokens as correct/incorrect.
- Reasoning mode (`run_reasoning`): use TransformerLens/TransformerBridge to inject custom hooks, extracting and aggregating token-level and step-level means.
- Baseline mode (`run_baseline`): stream the FineWeb dataset via HuggingFace `AutoModelForCausalLM`, extract hidden states, and calculate a 50% variance PCA.
- The `DiskBackedActivationStore` class: it handles sharding, buffering, and saving raw tensors and aligned metadata to disk.

"""



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import torch
import logging
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

# ==========================================
# Configuration & Setup
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

MODEL_SLUG = MODEL_NAME.split("/")[-1]

# --- Reasoning-mode config ---
REASONING_DATASET = "ProcessBench"
REASONING_DATASET_TAG = REASONING_DATASET.lower()
DATASET_FILE = os.path.join("reasoning_datasets", REASONING_DATASET, "dataset.jsonl")

# --- Baseline-mode config ---
BASELINE_DATASET_TAG = "fineweb"
NUM_SAMPLES = 20000
MAX_LENGTH = 1024
START_TOKEN_IDX = 5
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

# --- Output directory roots ---
LOG_DIR = os.path.join("logs", MODEL_SLUG, REASONING_DATASET_TAG)
VECTOR_DIR = os.path.join("reasoning_vectors", MODEL_SLUG, REASONING_DATASET_TAG)

# --- Directory for raw per-token activation shards ---
RAW_ACTIVATIONS_DIR = os.path.join(VECTOR_DIR, "raw_activations")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(RAW_ACTIVATIONS_DIR, exist_ok=True)

# --- Fully-qualified file paths ---
REASONING_LOG_FILE    = os.path.join(LOG_DIR,    f"reasoning_analysis_{MODEL_SLUG}_{REASONING_DATASET_TAG}_with_steps_avg_storage.log")
REASONING_OUTPUT_FILE = os.path.join(VECTOR_DIR, f"reasoning_vectors_{MODEL_SLUG}_{REASONING_DATASET_TAG}_with_steps_avg_storage.pt")

BASELINE_LOG_FILE    = os.path.join(LOG_DIR,    f"fineweb_baseline_{MODEL_SLUG}_{NUM_SAMPLES}_storage.log")
BASELINE_OUTPUT_FILE = os.path.join(VECTOR_DIR, f"fineweb_activations_{MODEL_SLUG}_{NUM_SAMPLES}_storage.pt")

# --- Memory management config ---
# Max number of token-rows to buffer in RAM before flushing to disk.
# At d_model=4096 and float16, each row is ~8 KB. 100k rows ≈ 800 MB per layer.
CHUNK_FLUSH_TOKENS = 100_000


# ==========================================
# Disk-Backed Activation Store
# ==========================================
class DiskBackedActivationStore:
    """Accumulates per-token activations in RAM and flushes to numbered
    shard files on disk when the buffer exceeds *max_tokens_in_ram*.

    After collection, call :meth:`finalize` to write an index file that
    maps every token row back to its sample, step, and correctness label.

    Storage layout on disk (one sub-dir per hook name)::

        raw_activations/
            blocks.18.hook_out/
                shard_000.pt   # (N, d_model) bfloat16
                shard_001.pt
                ...
            blocks.19.hook_out/
                ...
            token_metadata.pt  # list[dict] aligned row-for-row with shards
            index.pt           # offsets, counts, shard boundaries
    """

    def __init__(self, root_dir: str, hook_names: list[str], d_model: int,
                 max_tokens_in_ram: int = CHUNK_FLUSH_TOKENS,
                 dtype: torch.dtype = torch.bfloat16):
        self.root = Path(root_dir)
        self.hook_names = hook_names
        self.d_model = d_model
        self.max_tokens = max_tokens_in_ram
        self.dtype = dtype

        # Per-hook RAM buffer and shard counter
        self._buffers: dict[str, list[torch.Tensor]] = {n: [] for n in hook_names}
        self._buf_len = 0  # current number of token-rows buffered (same across hooks)
        self._shard_counts: dict[str, int] = {n: 0 for n in hook_names}
        self._total_rows = 0

        # Per-token metadata (aligned row-for-row with activations)
        self._meta_buffer: list[dict] = []

        # Per-sample boundary index: list of (start_row, num_tokens, sample_idx)
        self._sample_index: list[tuple[int, int, int]] = []

        # Create per-hook directories
        for name in hook_names:
            (self.root / self._safe_name(name)).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(hook_name: str) -> str:
        return hook_name.replace(".", "_")

    # ------------------------------------------------------------------
    def append(self, cache: dict[str, torch.Tensor],
               metadata_rows: list[dict], sample_idx: int):
        """Add all reasoning-token activations for one sample.

        Parameters
        ----------
        cache : dict  hook_name → (1, num_reasoning_tokens, d_model) CPU tensor
        metadata_rows : list[dict]  one dict per reasoning token
        sample_idx : int
        """
        num_tokens = len(metadata_rows)
        if num_tokens == 0:
            return

        start_row = self._total_rows

        for name in self.hook_names:
            # cache[name] shape: (1, T, d_model) — squeeze batch dim
            acts = cache[name].squeeze(0).to(self.dtype)  # (T, d_model)
            self._buffers[name].append(acts)

        self._meta_buffer.extend(metadata_rows)
        self._buf_len += num_tokens
        self._total_rows += num_tokens
        self._sample_index.append((start_row, num_tokens, sample_idx))

        if self._buf_len >= self.max_tokens:
            self._flush()

    # ------------------------------------------------------------------
    def _flush(self):
        """Write current RAM buffers to numbered shard files."""
        if self._buf_len == 0:
            return

        for name in self.hook_names:
            stacked = torch.cat(self._buffers[name], dim=0)  # (buf_len, d_model)
            shard_id = self._shard_counts[name]
            shard_path = self.root / self._safe_name(name) / f"shard_{shard_id:04d}.pt"
            torch.save(stacked, shard_path)
            self._shard_counts[name] += 1
            self._buffers[name].clear()

        # Flush metadata chunk
        meta_path = self.root / f"meta_shard_{self._shard_counts[self.hook_names[0]] - 1:04d}.pt"
        torch.save(self._meta_buffer, meta_path)
        self._meta_buffer = []

        logging.info(f"  [DiskStore] Flushed {self._buf_len} token-rows to shard "
                     f"{self._shard_counts[self.hook_names[0]] - 1}  "
                     f"(total so far: {self._total_rows})")
        self._buf_len = 0

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """Flush remaining buffer, write the index, return summary dict."""
        self._flush()

        index = {
            "total_token_rows": self._total_rows,
            "num_shards": self._shard_counts[self.hook_names[0]],
            "sample_index": self._sample_index,  # [(start_row, num_tokens, sample_idx), ...]
            "hook_names": self.hook_names,
            "d_model": self.d_model,
            "dtype": str(self.dtype),
        }
        torch.save(index, self.root / "index.pt")
        logging.info(f"  [DiskStore] Finalized: {self._total_rows} total token-rows "
                     f"across {index['num_shards']} shards.")
        return index


# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified activation analysis script.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["reasoning", "baseline"],
        help="Analysis mode: 'reasoning' uses ProcessBench + TransformerBridge, "
             "'baseline' uses FineWeb + AutoModelForCausalLM.",
    )
    return parser.parse_args()


# ==========================================
# Logging helper
# ==========================================
def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


# ==========================================
# ProcessBench helpers  (reasoning mode)
# ==========================================
def prepare_prompt_and_labels_processbench(sample, tokenizer):
    """Aligns ProcessBench steps with token positions.

    Returns
    -------
    input_ids : torch.Tensor  shape (1, seq_len)
    token_metadata : list[dict]  per-token {is_correct, step_idx}
    first_error_token_idx : int   (-1 if none)
    reasoning_start_token_idx : int
    """
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [{"start": 0, "end": len(full_text), "step_idx": -1, "is_correct": None}]

    first_error_step_idx = sample["label"]

    for idx, step in enumerate(sample["steps"]):
        start_char = len(full_text)
        step_text = f"Step {idx + 1}: {step}\n"
        full_text += step_text
        end_char = len(full_text)

        if first_error_step_idx == -1:
            is_correct = True
        else:
            is_correct = True if idx < first_error_step_idx else False

        char_regions.append(
            {"start": start_char, "end": end_char,
             "step_idx": idx, "is_correct": is_correct}
        )

    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets = encoding["offset_mapping"]

    token_metadata = []
    first_error_token_idx = -1
    reasoning_start_token_idx = -1

    for pos, (start, end) in enumerate(offsets):
        assigned_region = next(
            (r for r in char_regions if start >= r["start"] and end <= r["end"]), None
        )
        if assigned_region:
            step_idx = assigned_region["step_idx"]

            if reasoning_start_token_idx == -1 and step_idx == 0:
                reasoning_start_token_idx = pos

            if (
                first_error_token_idx == -1
                and step_idx == first_error_step_idx
                and first_error_step_idx != -1
            ):
                first_error_token_idx = pos

            token_metadata.append({"is_correct": assigned_region["is_correct"], "step_idx": step_idx})
        else:
            token_metadata.append({"is_correct": None, "step_idx": -1})

    return input_ids, token_metadata, first_error_token_idx, reasoning_start_token_idx


# ==========================================
# REASONING MODE
# ==========================================
def run_reasoning():
    setup_logging(REASONING_LOG_FILE)

    logging.info(f"[reasoning] Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    logging.info(f"Log  → {REASONING_LOG_FILE}")
    logging.info(f"Out  → {REASONING_OUTPUT_FILE}")
    logging.info(f"Raw  → {RAW_ACTIVATIONS_DIR}")

    from transformer_lens.model_bridge import TransformerBridge

    if torch.cuda.is_available():
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "All (Not Restricted)")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Hardware: GPU {gpu_idx} ({gpu_name})")
    else:
        logging.info("Hardware: CPU")

    logging.info("Loading model via TransformerBridge...")
    model = TransformerBridge.boot_transformers(MODEL_NAME)
    model.enable_compatibility_mode(disable_warnings=True, no_processing=True)
    model.eval()
    d_model = model.cfg.d_model

    resid_post_names = [f"blocks.{layer}.hook_out" for layer in TARGET_LAYERS]
    target_names_set = set(resid_post_names)
    # Keep a stable ordering for the disk store
    target_names_list = sorted(target_names_set)

    # -- Initialise disk-backed store for ALL per-token activations --
    disk_store = DiskBackedActivationStore(
        root_dir=RAW_ACTIVATIONS_DIR,
        hook_names=target_names_list,
        d_model=d_model,
        max_tokens_in_ram=CHUNK_FLUSH_TOKENS,
    )

    # -- Aggregation tensors (kept in RAM — these are just running sums) --
    running_sum_correct   = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}
    running_sum_incorrect = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}

    running_sum_correct_step   = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}
    running_sum_incorrect_step = {name: torch.zeros(d_model, device="cpu") for name in target_names_set}

    per_sample_layer_means      = {name: [] for name in target_names_set}
    per_sample_is_fully_correct = []

    count_correct   = 0
    count_incorrect = 0
    count_correct_steps   = 0
    count_incorrect_steps = 0

    # -- Load dataset --
    dataset = []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(dataset)} samples from {DATASET_FILE}.")
    except FileNotFoundError:
        logging.error(f"Dataset file '{DATASET_FILE}' not found. Run format_dataset.py first.")
        return

    # -- Forward passes --
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Forward Passes")):
            try:
                tokens, token_labels, first_error_token_idx, reasoning_start_token_idx = (
                    prepare_prompt_and_labels_processbench(sample, model.tokenizer)
                )
                tokens = tokens.to(model.cfg.device)

                cache = {}

                def reasoning_cache_hook(tensor, hook):
                    cache[hook.name] = tensor[:, reasoning_start_token_idx:, :].cpu()

                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(name, reasoning_cache_hook) for name in target_names_set],
                )

                reasoning_metadata = token_labels[reasoning_start_token_idx:]

                step_to_token_indices = defaultdict(list)

                # Build per-token metadata rows enriched with sample_idx
                token_meta_rows = []
                for pos, meta in enumerate(reasoning_metadata):
                    is_correct = meta["is_correct"]
                    step_idx = meta["step_idx"]

                    if step_idx >= 0:
                        step_to_token_indices[step_idx].append(pos)

                    if is_correct is True:
                        for name in target_names_set:
                            running_sum_correct[name] += cache[name][0, pos]
                        count_correct += 1
                    elif is_correct is False:
                        for name in target_names_set:
                            running_sum_incorrect[name] += cache[name][0, pos]
                        count_incorrect += 1

                    token_meta_rows.append({
                        "sample_idx": i,
                        "token_pos": pos,
                        "step_idx": step_idx,
                        "is_correct": is_correct,
                    })

                # ---- Write ALL per-token activations to disk ----
                disk_store.append(cache, token_meta_rows, sample_idx=i)

                # STEP-LEVEL LOGIC (unchanged)
                label = sample.get("label")
                if label == -1:
                    for s_idx, indices in step_to_token_indices.items():
                        if len(indices) > 0:
                            for name in target_names_set:
                                step_tokens = cache[name][0, indices, :]
                                step_mean = step_tokens.mean(dim=0)
                                running_sum_correct_step[name] += step_mean
                            count_correct_steps += 1
                else:
                    first_err_idx = label

                    for s_idx in range(first_err_idx):
                        if s_idx in step_to_token_indices:
                            indices = step_to_token_indices[s_idx]
                            if len(indices) > 0:
                                for name in target_names_set:
                                    step_tokens = cache[name][0, indices, :]
                                    step_mean = step_tokens.mean(dim=0)
                                    running_sum_correct_step[name] += step_mean
                                count_correct_steps += 1

                    if first_err_idx in step_to_token_indices:
                        indices = step_to_token_indices[first_err_idx]
                        if len(indices) > 0:
                            for name in target_names_set:
                                step_tokens = cache[name][0, indices, :]
                                step_mean = step_tokens.mean(dim=0)
                                running_sum_incorrect_step[name] += step_mean
                            count_incorrect_steps += 1

                # Per-sample mean across all reasoning tokens (kept for backward compat)
                for name in target_names_set:
                    sample_avg = cache[name].mean(dim=1).squeeze(0).clone()
                    per_sample_layer_means[name].append(sample_avg)

                is_perfect_sample = (
                    sample.get("label") == -1 and sample.get("final_answer_correct") is True
                )
                per_sample_is_fully_correct.append(is_perfect_sample)

                del cache, logits
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue

    # -- Finalize disk store (flush remaining buffer, write index) --
    store_index = disk_store.finalize()

    # -- Compute final vectors --
    logging.info("Computing mean vectors...")
    results = {"layers": {}, "metadata": {}}

    is_perfect_mask = torch.tensor(per_sample_is_fully_correct, dtype=torch.bool)

    for name in target_names_set:
        mean_correct_token = (
            running_sum_correct[name] / count_correct
            if count_correct > 0 else torch.zeros(d_model)
        )
        mean_incorrect_token = (
            running_sum_incorrect[name] / count_incorrect
            if count_incorrect > 0 else torch.zeros(d_model)
        )

        mean_correct_step = (
            running_sum_correct_step[name] / count_correct_steps
            if count_correct_steps > 0 else torch.zeros(d_model)
        )
        mean_incorrect_step = (
            running_sum_incorrect_step[name] / count_incorrect_steps
            if count_incorrect_steps > 0 else torch.zeros(d_model)
        )

        stacked_per_sample_means = torch.stack(per_sample_layer_means[name])

        mean_correct_samples = (
            stacked_per_sample_means[is_perfect_mask].mean(dim=0)
            if is_perfect_mask.any() else torch.zeros(d_model)
        )
        mean_incorrect_samples = (
            stacked_per_sample_means[~is_perfect_mask].mean(dim=0)
            if (~is_perfect_mask).any() else torch.zeros(d_model)
        )

        results["layers"][name] = {
            "mean_correct_token":         mean_correct_token,
            "mean_incorrect_token":       mean_incorrect_token,
            "reasoning_direction_token":  mean_correct_token - mean_incorrect_token,

            "mean_correct_step":          mean_correct_step,
            "mean_incorrect_step":        mean_incorrect_step,
            "reasoning_direction_step":   mean_correct_step - mean_incorrect_step,

            "per_sample_means":           stacked_per_sample_means,
            "mean_correct_samples":       mean_correct_samples,
            "mean_incorrect_samples":     mean_incorrect_samples,
            "reasoning_direction_sample": mean_correct_samples - mean_incorrect_samples,
        }

    results["metadata"] = {
        "model":                       MODEL_NAME,
        "target_layers":               TARGET_LAYERS,
        "count_correct_tokens":        count_correct,
        "count_incorrect_tokens":      count_incorrect,
        "count_correct_steps":         count_correct_steps,
        "count_incorrect_steps":       count_incorrect_steps,
        "total_successful_samples":    len(per_sample_is_fully_correct),
        "per_sample_is_fully_correct": is_perfect_mask,
        # Pointer to the raw activation store
        "raw_activations_dir":         RAW_ACTIVATIONS_DIR,
        "raw_activations_index":       store_index,
    }

    torch.save(results, REASONING_OUTPUT_FILE)
    logging.info(f"Saved reasoning vectors → {REASONING_OUTPUT_FILE}")
    logging.info(f"Saved raw activations  → {RAW_ACTIVATIONS_DIR}  "
                 f"({store_index['total_token_rows']} token-rows, "
                 f"{store_index['num_shards']} shards)")
    logging.info(f"Token counts  → Correct: {count_correct}, Incorrect: {count_incorrect}")
    logging.info(f"Step counts   → Correct: {count_correct_steps}, Incorrect: {count_incorrect_steps}")
    logging.info(
        f"Sample counts → Perfect: {is_perfect_mask.sum().item()}, "
        f"Flawed: {(~is_perfect_mask).sum().item()}"
    )


# ==========================================
# Utility: Load raw activations back from disk
# ==========================================
def load_raw_activations(raw_dir: str, hook_name: str,
                         sample_indices: list[int] | None = None) -> tuple[torch.Tensor, list[dict]]:
    """Convenience loader for downstream analysis.

    Parameters
    ----------
    raw_dir : path to the raw_activations directory
    hook_name : e.g. "blocks.22.hook_out"
    sample_indices : optional list of sample ids to filter; None = load all

    Returns
    -------
    activations : (N, d_model)  tensor of selected token rows
    metadata    : list[dict]    aligned per-row metadata
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)

    safe = hook_name.replace(".", "_")
    num_shards = index["num_shards"]

    # Build set of desired (start_row, num_tokens) ranges
    if sample_indices is not None:
        desired = set(sample_indices)
        ranges = [(s, n) for s, n, sid in index["sample_index"] if sid in desired]
    else:
        ranges = None  # load everything

    all_acts = []
    all_meta = []
    global_row = 0

    for shard_id in range(num_shards):
        acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt", weights_only=False)
        meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False)
        shard_len = acts.shape[0]

        if ranges is None:
            all_acts.append(acts)
            all_meta.extend(meta)
        else:
            # Select only rows belonging to requested samples
            mask = torch.zeros(shard_len, dtype=torch.bool)
            kept_meta = []
            for start, length in ranges:
                local_start = start - global_row
                local_end = local_start + length
                if 0 <= local_start < shard_len:
                    actual_end = min(local_end, shard_len)
                    mask[local_start:actual_end] = True
                    kept_meta.extend(meta[local_start:actual_end])
            if mask.any():
                all_acts.append(acts[mask])
                all_meta.extend(kept_meta)

        global_row += shard_len

    return torch.cat(all_acts, dim=0), all_meta


# ==========================================
# BASELINE MODE  (unchanged)
# ==========================================
def run_baseline():
    setup_logging(BASELINE_LOG_FILE)

    logging.info(f"[baseline] Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    logging.info(f"Log  → {BASELINE_LOG_FILE}")
    logging.info(f"Out  → {BASELINE_OUTPUT_FILE}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    logging.info("Loading tokenizer and model via AutoModelForCausalLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    logging.info(f"Loading {DATASET_NAME} (streaming)...")
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    layer_activations = {layer: [] for layer in TARGET_LAYERS}

    count = 0
    with tqdm(total=NUM_SAMPLES, desc="Forward passes") as pbar:
        for row in dataset:
            if count >= NUM_SAMPLES:
                break

            text = row["text"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
            ).to(model.device)

            seq_len = inputs["input_ids"].shape[1]
            if seq_len <= START_TOKEN_IDX + 10:
                continue

            try:
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                for layer in TARGET_LAYERS:
                    layer_hidden = hidden_states[layer]
                    avg_activation = layer_hidden[0, START_TOKEN_IDX:, :].mean(dim=0).cpu()
                    layer_activations[layer].append(avg_activation)

                count += 1
                pbar.update(1)

            except Exception as e:
                logging.error(f"Error processing sample {count}: {e}")
                continue

    logging.info("Stacking and saving activations...")
    results = {
        "layers": {},
        "metadata": {
            "model":           MODEL_NAME,
            "target_layers":   TARGET_LAYERS,
            "num_samples":     count,
            "dataset":         DATASET_NAME,
            "start_token_idx": START_TOKEN_IDX,
        },
    }

    pca_components = {}
    for layer in TARGET_LAYERS:
        stacked = torch.stack(layer_activations[layer])
        results["layers"][layer] = stacked

        X = stacked.to(torch.float32).numpy()
        pca = PCA(n_components=0.50, svd_solver='full')
        pca.fit(X)
        pca_components[layer] = torch.tensor(pca.components_, dtype=torch.float32)
        logging.info(f"  Layer {layer}: {len(pca.components_)} components explain 50% variance.")

    torch.save(results, BASELINE_OUTPUT_FILE)
    logging.info(f"Saved FineWeb activations → {BASELINE_OUTPUT_FILE}")

    pca_output_file = os.path.join(VECTOR_DIR, f"fineweb_pca_components_{MODEL_SLUG}_{NUM_SAMPLES}.pt")
    torch.save({"layers": pca_components, "metadata": results["metadata"]}, pca_output_file)
    logging.info(f"Saved FineWeb PCA components → {pca_output_file}")


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    if args.type == "reasoning":
        run_reasoning()
    else:
        run_baseline()