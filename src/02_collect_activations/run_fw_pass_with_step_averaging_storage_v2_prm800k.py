"""
Script to run forward passes and extract layer-wise activations for the model in two distinct modes: 'reasoning' and 'baseline'. 
- In 'reasoning' mode, it aligns tokens with logical steps from the ProcessBench or Math-Shepherd dataset to compute mean activation vectors for correct and incorrect reasoning tokens/steps. 

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

Note: we DO save the raw activation value for each token

Structure:
- Arguments: can specifiy '--type reasoning' and '--type baseline', and '--dataset processbench' or '--dataset math-shepherd'.
- Helper function for reasoning dataset ProcessBench: `prepare_prompt_and_labels_processbench` aligns character offsets to specific reasoning steps to label tokens as correct/incorrect.
- Helper function for Math-Shepherd: `prepare_prompt_and_labels_mathshepherd` uses per-step boolean labels directly.
- Reasoning mode (`run_reasoning`): use TransformerLens/TransformerBridge to inject custom hooks, extracting and aggregating token-level and step-level means.
- Baseline mode (`run_baseline`): stream the FineWeb dataset via HuggingFace `AutoModelForCausalLM`, extract hidden states, and calculate a 50% variance PCA.
- The `DiskBackedActivationStore` class: it handles sharding, buffering, and saving raw tensors and aligned metadata to disk.

Per-token metadata always includes `problem_id` (taken from sample["problem_id"], falling back to str(sample_idx))
so every stored token row can be traced back to its source dataset row without needing to re-read the JSONL.
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
            meta_shard_0000.pt  # list[dict] aligned row-for-row with branch shards
            index.pt            # offsets, counts, shard boundaries

        PRM800K dedup layout adds:
            prefix_blocks.18.hook_out/
                shard_000.pt   # prefix activations stored once per dataset row
                ...
            prefix_meta/
                shard_0000.pt  # list[dict] for prefix tokens
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
        # For PRM800K branches, entries are 4-tuples: (start_row, num_tokens, sample_idx, prefix_id)
        self._sample_index: list[tuple] = []

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
        metadata_rows : list[dict]  one dict per reasoning token; must include
                        at minimum: sample_idx, token_pos, step_idx, is_correct,
                        problem_id.
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
    # Prefix-deduplication API (PRM800K)
    # ------------------------------------------------------------------
    def store_prefix(self, cache: dict[str, torch.Tensor],
                     metadata_rows: list[dict]) -> int:
        """Save a shared prefix block to a dedicated prefix shard.

        Each unique prefix is written once.  Returns an integer ``prefix_id``
        that callers pass to :meth:`append_branch` for every branch that
        shares this prefix.

        Parameters
        ----------
        cache : hook_name → (1, T_prefix, d_model) CPU tensor
        metadata_rows : list[dict]  per-token metadata for the prefix tokens;
                        must include problem_id.
        """
        if not hasattr(self, "_prefix_buffers"):
            # Lazy init — only needed for prm800k dedup path
            self._prefix_buffers: dict[str, list[torch.Tensor]] = {n: [] for n in self.hook_names}
            self._prefix_meta_buffer: list[dict] = []
            self._prefix_shard_counts: dict[str, int] = {n: 0 for n in self.hook_names}
            self._prefix_total_rows: int = 0
            self._prefix_index: list[tuple[int, int]] = []  # (start_row, num_tokens)
            for name in self.hook_names:
                (self.root / ("prefix_" + self._safe_name(name))).mkdir(parents=True, exist_ok=True)
            (self.root / "prefix_meta").mkdir(parents=True, exist_ok=True)

        num_tokens = len(metadata_rows)
        prefix_id = len(self._prefix_index)
        start_row = self._prefix_total_rows

        for name in self.hook_names:
            acts = cache[name].squeeze(0).to(self.dtype)
            self._prefix_buffers[name].append(acts)

        self._prefix_meta_buffer.extend(metadata_rows)
        self._prefix_total_rows += num_tokens
        self._prefix_index.append((start_row, num_tokens))

        # Flush prefix buffer when large enough
        if self._prefix_total_rows >= self.max_tokens:
            self._flush_prefix()

        return prefix_id

    def _flush_prefix(self):
        """Write prefix RAM buffers to disk."""
        if not hasattr(self, "_prefix_buffers") or not self._prefix_buffers[self.hook_names[0]]:
            return
        for name in self.hook_names:
            stacked = torch.cat(self._prefix_buffers[name], dim=0)
            sid = self._prefix_shard_counts[name]
            path = self.root / ("prefix_" + self._safe_name(name)) / f"shard_{sid:04d}.pt"
            torch.save(stacked, path)
            self._prefix_shard_counts[name] += 1
            self._prefix_buffers[name].clear()
        sid = self._prefix_shard_counts[self.hook_names[0]] - 1
        torch.save(self._prefix_meta_buffer,
                   self.root / "prefix_meta" / f"shard_{sid:04d}.pt")
        self._prefix_meta_buffer = []

    def append_branch(self, cache: dict[str, torch.Tensor],
                      metadata_rows: list[dict],
                      sample_idx: int, prefix_id: int):
        """Append branch-only activations and record the prefix_id pointer.

        The on-disk row for this sample only contains the *branch* tokens.
        The shared prefix is stored separately and referenced via ``prefix_id``.
        The ``sample_index`` entry is a 4-tuple so that
        :func:`load_raw_activations` can reconstruct the full sequence.

        Parameters
        ----------
        cache : hook_name → (1, T_branch, d_model) CPU tensor
        metadata_rows : list[dict]  per-token metadata for branch tokens only;
                        must include problem_id.
        sample_idx : int
        prefix_id : int  returned by a prior :meth:`store_prefix` call
        """
        num_tokens = len(metadata_rows)
        if num_tokens == 0:
            return

        start_row = self._total_rows

        for name in self.hook_names:
            acts = cache[name].squeeze(0).to(self.dtype)
            self._buffers[name].append(acts)

        self._meta_buffer.extend(metadata_rows)
        self._buf_len += num_tokens
        self._total_rows += num_tokens
        # 4-tuple with prefix pointer
        self._sample_index.append((start_row, num_tokens, sample_idx, prefix_id))

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
        if hasattr(self, "_prefix_buffers"):
            self._flush_prefix()

        index = {
            "total_token_rows": self._total_rows,
            "num_shards": self._shard_counts[self.hook_names[0]],
            # Each entry is either (start_row, num_tokens, sample_idx)
            # or for PRM800K branches: (start_row, num_tokens, sample_idx, prefix_id)
            "sample_index": self._sample_index,
            "hook_names": self.hook_names,
            "d_model": self.d_model,
            "dtype": str(self.dtype),
            # Prefix dedup fields (present only for prm800k; None otherwise)
            "prefix_index": getattr(self, "_prefix_index", None),
            "num_prefix_shards": (
                self._prefix_shard_counts[self.hook_names[0]]
                if hasattr(self, "_prefix_shard_counts") else 0
            ),
        }
        torch.save(index, self.root / "index.pt")
        logging.info(f"  [DiskStore] Finalized: {self._total_rows} total token-rows "
                     f"across {index['num_shards']} shards.")
        if index["prefix_index"]:
            logging.info(f"  [DiskStore] Prefix blocks: {len(index['prefix_index'])} unique prefixes, "
                         f"{index['num_prefix_shards']} prefix shards.")
        return index


# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified activation analysis script.")

    parser.add_argument(
        "--dataset",
        default="processbench",
        choices=["processbench", "math-shepherd", "prm800k"],
        help="Reasoning dataset to use (only relevant when --type=reasoning). "
             "Default: processbench.",
    )
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="Path to the dataset .jsonl file. If not provided, defaults to "
             "reasoning_datasets/<dataset>/dataset.jsonl for ProcessBench or "
             "/home/ines/Reasoning-activations/reasoning_datasets/math_shepherd/math_shepherd_dataset_3000samples.jsonl"
             "for Math-Shepherd.",
    )
    return parser.parse_args()


def resolve_dataset_config(args):
    """Return (dataset_tag, dataset_file) based on parsed args."""
    if args.dataset == "processbench":
        tag = "processbench"
        default_file = os.path.join("reasoning_datasets", "ProcessBench", "dataset.jsonl")
    elif args.dataset == "math-shepherd":
        tag = "math-shepherd"
        default_file = "/home/ines/Reasoning-activations/reasoning_datasets/math_shepherd/math_shepherd_dataset_3000samples.jsonl"
    else:  # prm800k
        tag = "prm800k"
        # default_file = "/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned_multiple_traj.jsonl"
        default_file = "/home/ines/Reasoning-activations/reasoning_datasets/prm800k/prm800k_phase2_test_cleaned_w_problem_ids.jsonl"

    dataset_file = args.dataset_file if args.dataset_file else default_file
    return tag, dataset_file


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
# Math-Shepherd helpers  (reasoning mode)
# ==========================================
def prepare_prompt_and_labels_mathshepherd(sample, tokenizer):
    """Aligns Math-Shepherd steps with token positions using per-step labels.

    Math-Shepherd provides a boolean `step_labels` list, so each step has its
    own independent correctness label (unlike ProcessBench where correctness
    is derived from first-error-index).

    Returns
    -------
    input_ids : torch.Tensor  shape (1, seq_len)
    token_metadata : list[dict]  per-token {is_correct, step_idx}
    first_error_token_idx : int   (-1 if none)
    reasoning_start_token_idx : int
    """
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [{"start": 0, "end": len(full_text), "step_idx": -1, "is_correct": None}]

    step_labels = sample["step_labels"]  # list[bool], per-step correctness

    for idx, step in enumerate(sample["steps"]):
        start_char = len(full_text)
        step_text = f"Step {idx + 1}: {step}\n"
        full_text += step_text
        end_char = len(full_text)

        # Use the per-step label directly
        is_correct = step_labels[idx] if idx < len(step_labels) else None

        char_regions.append(
            {"start": start_char, "end": end_char,
             "step_idx": idx, "is_correct": is_correct}
        )

    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets = encoding["offset_mapping"]

    # Find the first error step index for first_error_token_idx tracking
    first_error_step_idx = sample.get("label", -1)

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
                and step_idx >= 0
                and assigned_region["is_correct"] is False
            ):
                first_error_token_idx = pos

            token_metadata.append({"is_correct": assigned_region["is_correct"], "step_idx": step_idx})
        else:
            token_metadata.append({"is_correct": None, "step_idx": -1})

    return input_ids, token_metadata, first_error_token_idx, reasoning_start_token_idx


# ==========================================
# PRM800K helpers  (reasoning mode)
# ==========================================
def prepare_prompt_and_labels_prm800k(sample, tokenizer):
    """Aligns PRM800K steps with token positions, treating every completion
    within each step as an independently-labelled sub-step.

    PRM800K structure:
        sample["problem"]  : str
        sample["steps"]    : list of step dicts, each with:
            step["completions"] : list of completion dicts, each with:
                completion["text"]   : str
                completion["rating"] : int  (-1, 0, or +1)

    Each completion is appended to the full text and assigned its own
    char region with:
        step_label  : raw rating value (-1, 0, +1)
        is_correct  : True if rating in {0, +1}, False if rating == -1

    The step_idx encodes both the parent step and the completion index as
    a flat counter so that downstream step-level averaging treats each
    completion as its own independent unit.

    Returns
    -------
    input_ids               : torch.Tensor  shape (1, seq_len)
    token_metadata          : list[dict]    per-token dicts with keys:
                                  is_correct, step_idx, step_label
    first_error_token_idx   : int   (-1 if no incorrect completion found)
    reasoning_start_token_idx : int
    """
    full_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    char_regions = [
        {"start": 0, "end": len(full_text),
         "step_idx": -1, "is_correct": None, "step_label": None}
    ]

    flat_step_idx = 0          # monotonically increasing across all completions

    for step in sample["steps"]:
        for completion in step.get("completions", []):
            text    = completion.get("text", "")
            rating  = completion.get("rating", None)

            # rating → is_correct
            if rating == -1:
                is_correct = False
            elif rating in (0, 1):
                is_correct = True
            else:
                is_correct = None   # flagged / missing rating

            start_char = len(full_text)
            full_text += text + "\n"
            end_char = len(full_text)

            char_regions.append({
                "start":      start_char,
                "end":        end_char,
                "step_idx":   flat_step_idx,
                "is_correct": is_correct,
                "step_label": rating,        # raw -1 / 0 / +1
            })
            flat_step_idx += 1

    encoding = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0)
    offsets   = encoding["offset_mapping"]

    token_metadata            = []
    first_error_token_idx     = -1
    reasoning_start_token_idx = -1

    for pos, (start, end) in enumerate(offsets):
        assigned_region = next(
            (r for r in char_regions if start >= r["start"] and end <= r["end"]),
            None,
        )
        if assigned_region:
            step_idx   = assigned_region["step_idx"]
            is_correct = assigned_region["is_correct"]

            # Mark the first reasoning token (first completion, flat_step_idx == 0)
            if reasoning_start_token_idx == -1 and step_idx == 0:
                reasoning_start_token_idx = pos

            # Mark the first token of the first incorrect completion
            if (
                first_error_token_idx == -1
                and step_idx >= 0
                and is_correct is False
            ):
                first_error_token_idx = pos

            token_metadata.append({
                "is_correct": is_correct,
                "step_idx":   step_idx,
                "step_label": assigned_region["step_label"],
            })
        else:
            token_metadata.append({
                "is_correct": None,
                "step_idx":   -1,
                "step_label": None,
            })

    return input_ids, token_metadata, first_error_token_idx, reasoning_start_token_idx


# ==========================================
# PRM800K branch-dedup helper
# ==========================================
def prepare_prm800k_branches(sample, tokenizer):
    """Split a PRM800K sample into a shared prefix and per-branch suffixes.

    For a sample where the first step with multiple completions is at step N,
    the shared prefix is: problem header + steps[0..N-1] first completion.
    Each branch is one of the completions at step N, followed by the remaining
    steps --- which are NONE to be confirmed
    (each using their first completion only, since branches diverge here).

    Returns
    -------
    prefix_ids   : torch.Tensor  shape (1, T_prefix)
    prefix_meta  : list[dict]    per-token metadata for the prefix
    branches     : list[dict] with keys:
        "input_ids"    : torch.Tensor shape (1, T_branch)   branch-only tokens
        "token_meta"   : list[dict]   per-token metadata (step_idx continues from prefix)
        "rating"       : int          completion rating (-1, 0, +1)
        "is_correct"   : bool|None
        "branch_label" : str          human-readable "step{step_i}_comp{comp_j}"
    reasoning_start_token_idx : int   index into the *prefix* token sequence
    """
    steps = sample["steps"]

    # --- Find first branching step (first step with >1 completion) ---
    branch_step_i = None
    for si, step in enumerate(steps):
        if len(step.get("completions", [])) > 1:
            branch_step_i = si
            break

    # If no branching step, fall back: treat the whole sample as one branch
    # (prefix = problem header only, branch = full reasoning sequence)
    if branch_step_i is None:
        ids, meta, first_err, rs = prepare_prompt_and_labels_prm800k(sample, tokenizer)
        branch_meta = meta[rs:]
        comp = steps[-1]["completions"][0] if steps else {"rating": None}
        rating = comp.get("rating", None)
        is_correct = True if rating in (0, 1) else (False if rating == -1 else None)
        return (
            ids[:, :rs],       # prefix = tokens before reasoning_start
            meta[:rs],
            [{
                "input_ids":    ids[:, rs:],
                "token_meta":   branch_meta,
                "rating":       rating,
                "is_correct":   is_correct,
                "branch_label": "no_branch",
            }],
            rs,
        )

    # --- Build shared prefix text: problem header + steps 0..branch_step_i-1 ---
    prefix_text = "Problem:\n" + sample["problem"] + "\n\nReasoning:\n"
    prefix_char_regions = [
        {"start": 0, "end": len(prefix_text), "step_idx": -1,
         "is_correct": None, "step_label": None}
    ]
    flat_step_idx = 0  # shared across prefix and branches for consistent step_idx
    for si in range(branch_step_i):
        comp = steps[si]["completions"][0]   # always first completion for prefix
        text = comp.get("text", "")
        rating = comp.get("rating", None)
        is_correct = True if rating in (0, 1) else (False if rating == -1 else None)
        start_char = len(prefix_text)
        prefix_text += text + "\n"
        prefix_char_regions.append({
            "start": start_char, "end": len(prefix_text),
            "step_idx": flat_step_idx, "is_correct": is_correct, "step_label": rating,
        })
        flat_step_idx += 1

    # Tokenize prefix
    prefix_enc = tokenizer(prefix_text, return_offsets_mapping=True)
    prefix_ids = torch.tensor(prefix_enc["input_ids"]).unsqueeze(0)
    prefix_offsets = prefix_enc["offset_mapping"]

    prefix_meta = []
    reasoning_start_token_idx = -1
    for pos, (start, end) in enumerate(prefix_offsets):
        region = next((r for r in prefix_char_regions
                       if start >= r["start"] and end <= r["end"]), None)
        if region:
            si = region["step_idx"]
            if reasoning_start_token_idx == -1 and si == 0:
                reasoning_start_token_idx = pos
            prefix_meta.append({"is_correct": region["is_correct"],
                                 "step_idx":   si,
                                 "step_label": region["step_label"]})
        else:
            prefix_meta.append({"is_correct": None, "step_idx": -1, "step_label": None})

    # --- Build one branch per completion at branch_step_i ---
    branches = []
    branch_step_completions = steps[branch_step_i].get("completions", [])
    for comp_j, comp in enumerate(branch_step_completions):
        branch_text = ""
        branch_char_regions = []
        local_flat = flat_step_idx   # continues from prefix

        # This completion (the branching one)
        text = comp.get("text", "")
        rating = comp.get("rating", None)
        is_correct = True if rating in (0, 1) else (False if rating == -1 else None)
        start_char = len(branch_text)
        branch_text += text + "\n"
        branch_char_regions.append({
            "start": start_char, "end": len(branch_text),
            "step_idx": local_flat, "is_correct": is_correct, "step_label": rating,
        })
        local_flat += 1

        # Remaining steps after branch_step_i — use first completion only
        for si in range(branch_step_i + 1, len(steps)):
            comps = steps[si].get("completions", [])
            if not comps:
                continue
            c = comps[0]
            t = c.get("text", "")
            r = c.get("rating", None)
            ic = True if r in (0, 1) else (False if r == -1 else None)
            sc = len(branch_text)
            branch_text += t + "\n"
            branch_char_regions.append({
                "start": sc, "end": len(branch_text),
                "step_idx": local_flat, "is_correct": ic, "step_label": r,
            })
            local_flat += 1

        # Tokenize branch suffix
        branch_enc = tokenizer(branch_text, return_offsets_mapping=True)
        branch_ids = torch.tensor(branch_enc["input_ids"]).unsqueeze(0)
        branch_offsets = branch_enc["offset_mapping"]

        branch_meta = []
        for pos, (start, end) in enumerate(branch_offsets):
            region = next((r for r in branch_char_regions
                           if start >= r["start"] and end <= r["end"]), None)
            if region:
                branch_meta.append({"is_correct": region["is_correct"],
                                    "step_idx":   region["step_idx"],
                                    "step_label": region["step_label"]})
            else:
                branch_meta.append({"is_correct": None, "step_idx": -1, "step_label": None})

        branches.append({
            "input_ids":    branch_ids,
            "token_meta":   branch_meta,
            "rating":       rating,
            "is_correct":   is_correct,
            "branch_label": f"step{branch_step_i}_comp{comp_j}",
        })

    return prefix_ids, prefix_meta, branches, reasoning_start_token_idx


# ==========================================
# REASONING MODE
# ==========================================
def run_reasoning(dataset_tag, dataset_file):
    # --- Build paths from dataset tag ---
    log_dir = os.path.join("logs", MODEL_SLUG, dataset_tag)
    vector_dir = os.path.join("reasoning_vectors", MODEL_SLUG, dataset_tag)
    raw_activations_dir = os.path.join(vector_dir, "raw_activations")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(raw_activations_dir, exist_ok=True)

    reasoning_log_file = os.path.join(
        log_dir, f"reasoning_analysis_{MODEL_SLUG}_{dataset_tag}_with_steps_avg_storage.log"
    )
    reasoning_output_file = os.path.join(
        vector_dir, f"reasoning_vectors_{MODEL_SLUG}_{dataset_tag}_with_steps_avg_storage.pt"
    )

    setup_logging(reasoning_log_file)

    logging.info(f"[reasoning] Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    logging.info(f"[reasoning] Dataset: {dataset_tag} | File: {dataset_file}")
    logging.info(f"Log  → {reasoning_log_file}")
    logging.info(f"Out  → {reasoning_output_file}")
    logging.info(f"Raw  → {raw_activations_dir}")

    # --- Choose the prompt-builder based on dataset ---
    if dataset_tag == "processbench":
        prepare_fn = prepare_prompt_and_labels_processbench
    elif dataset_tag == "math-shepherd":
        prepare_fn = prepare_prompt_and_labels_mathshepherd
    else:  # prm800k
        prepare_fn = prepare_prompt_and_labels_prm800k

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
        root_dir=raw_activations_dir,
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
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(dataset)} samples from {dataset_file}.")
    except FileNotFoundError:
        logging.error(f"Dataset file '{dataset_file}' not found.")
        return

    # -- Forward passes --
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Forward Passes")):
            try:
                # problem_id is used in every per-token metadata row so stored
                # activations can be traced back to the source dataset row.
                # i think only present in prm800k dataset, be careful
                problem_id = sample.get("problem_id", str(i))

                # ============================================================
                # PRM800K: deduplicated prefix + per-branch forward passes
                # ============================================================
                if dataset_tag == "prm800k":
                    prefix_ids, prefix_meta, branches, reasoning_start_token_idx = (
                        prepare_prm800k_branches(sample, model.tokenizer)
                    )

                    # --- Run forward pass for shared prefix ---
                    prefix_cache = {}
                    def prefix_hook(tensor, hook):
                        prefix_cache[hook.name] = tensor[:, reasoning_start_token_idx:, :].cpu()

                    prefix_tokens = prefix_ids.to(model.cfg.device)
                    model.run_with_hooks(
                        prefix_tokens,
                        fwd_hooks=[(name, prefix_hook) for name in target_names_set],
                    )
                    del prefix_tokens

                    # Prefix metadata rows (only reasoning portion), with problem_id
                    prefix_reasoning_meta = prefix_meta[reasoning_start_token_idx:]
                    prefix_meta_rows = [
                        {
                            "problem_id": problem_id,
                            "sample_idx": i,
                            "token_pos":  pos,
                            "step_idx":   m["step_idx"],
                            "is_correct": m["is_correct"],
                        }
                        for pos, m in enumerate(prefix_reasoning_meta)
                    ]

                    # Accumulate prefix token stats (counted once, shared by all branches)
                    for pos, m in enumerate(prefix_reasoning_meta):
                        ic = m["is_correct"]
                        if ic is True:
                            for name in target_names_set:
                                running_sum_correct[name] += prefix_cache[name][0, pos]
                            count_correct += 1
                        elif ic is False:
                            for name in target_names_set:
                                running_sum_incorrect[name] += prefix_cache[name][0, pos]
                            count_incorrect += 1

                    # Step-level stats for prefix steps (counted once, not once per branch)
                    prefix_step_to_tok = defaultdict(list)
                    for pos, m in enumerate(prefix_reasoning_meta):
                        if m["step_idx"] >= 0:
                            prefix_step_to_tok[m["step_idx"]].append(pos)
                    for s_idx, indices in prefix_step_to_tok.items():
                        if not indices:
                            continue
                        step_is_correct = prefix_reasoning_meta[indices[0]]["is_correct"]
                        if step_is_correct is None:
                            continue
                        for name in target_names_set:
                            step_mean = prefix_cache[name][0, indices, :].mean(dim=0)
                            if step_is_correct:
                                running_sum_correct_step[name] += step_mean
                            else:
                                running_sum_incorrect_step[name] += step_mean
                        if step_is_correct:
                            count_correct_steps += 1
                        else:
                            count_incorrect_steps += 1

                    # Store prefix block ONCE on disk; get prefix_id
                    prefix_id = disk_store.store_prefix(prefix_cache, prefix_meta_rows)

                    # --- Run one forward pass per branch (prefix in context) ---
                    # Concatenate prefix + branch so the model attends over the full
                    # context, but only cache the last T_branch token positions so
                    # prefix activations are stored only once (via store_prefix above).
                    prefix_ids_gpu = prefix_ids.to(model.cfg.device)
                    for b_idx, branch in enumerate(branches):
                        branch_len = branch["input_ids"].shape[1]
                        full_ids = torch.cat(
                            [prefix_ids_gpu, branch["input_ids"].to(model.cfg.device)], dim=1
                        )
                        branch_cache = {}

                        def branch_hook(tensor, hook):
                            # Slice only the branch token positions from the full sequence
                            branch_cache[hook.name] = tensor[:, -branch_len:, :].cpu()

                        logits = model.run_with_hooks(
                            full_ids,
                            fwd_hooks=[(name, branch_hook) for name in target_names_set],
                        )
                        del full_ids, logits

                        branch_meta = branch["token_meta"]
                        branch_meta_rows = [
                            {
                                "problem_id":   problem_id,
                                "sample_idx":   i,
                                "branch_idx":   b_idx,
                                "branch_label": branch["branch_label"],
                                "token_pos":    pos,
                                "step_idx":     m["step_idx"],
                                "is_correct":   m["is_correct"],
                            }
                            for pos, m in enumerate(branch_meta)
                        ]

                        # Accumulate branch token stats
                        for pos, m in enumerate(branch_meta):
                            ic = m["is_correct"]
                            if ic is True:
                                for name in target_names_set:
                                    running_sum_correct[name] += branch_cache[name][0, pos]
                                count_correct += 1
                            elif ic is False:
                                for name in target_names_set:
                                    running_sum_incorrect[name] += branch_cache[name][0, pos]
                                count_incorrect += 1

                        # Step-level stats for branch steps
                        branch_step_to_tok = defaultdict(list)
                        for pos, m in enumerate(branch_meta):
                            if m["step_idx"] >= 0:
                                branch_step_to_tok[m["step_idx"]].append(pos)
                        for s_idx, indices in branch_step_to_tok.items():
                            if not indices:
                                continue
                            step_is_correct = branch_meta[indices[0]]["is_correct"]
                            if step_is_correct is None:
                                continue
                            for name in target_names_set:
                                step_mean = branch_cache[name][0, indices, :].mean(dim=0)
                                if step_is_correct:
                                    running_sum_correct_step[name] += step_mean
                                else:
                                    running_sum_incorrect_step[name] += step_mean
                            if step_is_correct:
                                count_correct_steps += 1
                            else:
                                count_incorrect_steps += 1

                        # Per-sample mean: prefix reasoning tokens + branch tokens combined
                        for name in target_names_set:
                            combined = torch.cat([
                                prefix_cache[name][0],   # (T_prefix_reasoning, d_model)
                                branch_cache[name][0],   # (T_branch, d_model)
                            ], dim=0)
                            sample_avg = combined.mean(dim=0)
                            per_sample_layer_means[name].append(sample_avg)

                        # is_perfect: all prefix steps correct AND branch completion correct
                        prefix_all_correct = all(
                            m["is_correct"] is not False
                            for m in prefix_reasoning_meta if m["step_idx"] >= 0
                        )
                        branch_is_correct = branch["is_correct"]
                        is_perfect_sample = prefix_all_correct and branch_is_correct is True
                        per_sample_is_fully_correct.append(is_perfect_sample)

                        # Store branch activations pointing to shared prefix
                        disk_store.append_branch(branch_cache, branch_meta_rows,
                                                 sample_idx=i, prefix_id=prefix_id)
                        del branch_cache

                    del prefix_ids_gpu, prefix_cache
                    torch.cuda.empty_cache()

                # ============================================================
                # ProcessBench / Math-Shepherd: original single-pass logic
                # ============================================================
                else:
                    tokens, token_labels, first_error_token_idx, reasoning_start_token_idx = (
                        prepare_fn(sample, model.tokenizer)
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

                    # Build per-token metadata rows enriched with problem_id and sample_idx
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
                            "problem_id": problem_id,
                            "sample_idx": i,
                            "token_pos":  pos,
                            "step_idx":   step_idx,
                            "is_correct": is_correct,
                        })

                    # ---- Write ALL per-token activations to disk ----
                    disk_store.append(cache, token_meta_rows, sample_idx=i)

                    # ---- STEP-LEVEL LOGIC ----
                    if dataset_tag == "processbench":
                        # ProcessBench: label == -1 means all correct; otherwise
                        # steps < label are correct, step == label is the first error.
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
                    else:
                        # Math-Shepherd: use per-step labels directly.
                        # Each step is independently labeled correct/incorrect.
                        step_labels = sample["step_labels"]
                        for s_idx, indices in step_to_token_indices.items():
                            if len(indices) == 0:
                                continue
                            if s_idx >= len(step_labels):
                                continue

                            step_is_correct = step_labels[s_idx]

                            for name in target_names_set:
                                step_tokens = cache[name][0, indices, :]
                                step_mean = step_tokens.mean(dim=0)
                                if step_is_correct:
                                    running_sum_correct_step[name] += step_mean
                                else:
                                    running_sum_incorrect_step[name] += step_mean

                            if step_is_correct:
                                count_correct_steps += 1
                            else:
                                count_incorrect_steps += 1

                    # Per-sample mean across all reasoning tokens (kept for backward compat)
                    for name in target_names_set:
                        sample_avg = cache[name].mean(dim=1).squeeze(0).clone()
                        per_sample_layer_means[name].append(sample_avg)

                    if dataset_tag == "processbench":
                        # ProcessBench: a sample is "perfect" if label==-1 AND final_answer_correct
                        is_perfect_sample = (
                            sample.get("label") == -1 and sample.get("final_answer_correct") is True
                        )
                    else:
                        # Math-Shepherd: a sample is "perfect" if all step_labels are True
                        is_perfect_sample = all(sample.get("step_labels", []))

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
        "dataset":                     dataset_tag,
        "dataset_file":                dataset_file,
        "target_layers":               TARGET_LAYERS,
        "count_correct_tokens":        count_correct,
        "count_incorrect_tokens":      count_incorrect,
        "count_correct_steps":         count_correct_steps,
        "count_incorrect_steps":       count_incorrect_steps,
        "total_successful_samples":    len(per_sample_is_fully_correct),
        "per_sample_is_fully_correct": is_perfect_mask,
        # Pointer to the raw activation store
        "raw_activations_dir":         raw_activations_dir,
        "raw_activations_index":       store_index,
    }

    torch.save(results, reasoning_output_file)
    logging.info(f"Saved reasoning vectors → {reasoning_output_file}")
    logging.info(f"Saved raw activations  → {raw_activations_dir}  "
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
                         sample_indices: list[int] | None = None,
                         include_prefix: bool = True) -> tuple[torch.Tensor, list[dict]]:
    """Convenience loader for downstream analysis.

    Each returned metadata dict contains ``problem_id`` so rows can be
    traced back to their source dataset entry without re-reading the JSONL.

    For PRM800K (dedup mode) each branch entry in ``sample_index`` has the form
    ``(start_row, num_branch_tokens, sample_idx, prefix_id)``.  When
    ``include_prefix=True`` (default) the returned tensor for each branch is the
    *concatenation* of the shared prefix tokens and the branch tokens, exactly
    as if the full sequence had been stored.  Set ``include_prefix=False`` to
    load branch tokens only (useful when you only care about the new step).

    Parameters
    ----------
    raw_dir : path to the raw_activations directory
    hook_name : e.g. "blocks.22.hook_out"
    sample_indices : optional list of sample ids to filter; None = load all
    include_prefix : whether to prepend the shared prefix activations (prm800k only)

    Returns
    -------
    activations : (N, d_model)  tensor of selected token rows
    metadata    : list[dict]    aligned per-row metadata (includes problem_id)
    """
    raw_dir = Path(raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)

    safe = hook_name.replace(".", "_")
    num_shards = index["num_shards"]
    sample_index = index["sample_index"]
    prefix_index = index.get("prefix_index")   # None for non-prm800k datasets
    num_prefix_shards = index.get("num_prefix_shards", 0)

    has_prefix_dedup = (prefix_index is not None and num_prefix_shards > 0)

    # ---- Filter sample_index entries ----
    if sample_indices is not None:
        desired = set(sample_indices)
        entries = [e for e in sample_index if e[2] in desired]
    else:
        entries = list(sample_index)

    # ---- Non-dedup path (ProcessBench / Math-Shepherd) ----
    if not has_prefix_dedup:
        all_acts, all_meta = [], []
        global_row = 0
        ranges = {(e[0], e[1]) for e in entries}   # set of (start, length)
        for shard_id in range(num_shards):
            acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt", weights_only=False)
            meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False)
            shard_len = acts.shape[0]
            if not ranges:  # load everything if no filter
                all_acts.append(acts)
                all_meta.extend(meta)
            else:
                mask = torch.zeros(shard_len, dtype=torch.bool)
                kept_meta = []
                for start, length in ranges:
                    ls = start - global_row
                    le = ls + length
                    if 0 <= ls < shard_len:
                        ae = min(le, shard_len)
                        mask[ls:ae] = True
                        kept_meta.extend(meta[ls:ae])
                if mask.any():
                    all_acts.append(acts[mask])
                    all_meta.extend(kept_meta)
            global_row += shard_len
        return torch.cat(all_acts, dim=0), all_meta

    # ---- PRM800K dedup path ----
    # Load all prefix activations into memory (one block per dataset row)
    prefix_safe = "prefix_" + safe
    p_all_acts, p_all_meta = [], []
    for shard_id in range(num_prefix_shards):
        p_all_acts.append(torch.load(
            raw_dir / prefix_safe / f"shard_{shard_id:04d}.pt", weights_only=False))
        p_all_meta.extend(torch.load(
            raw_dir / "prefix_meta" / f"shard_{shard_id:04d}.pt", weights_only=False))
    prefix_acts_flat = torch.cat(p_all_acts, dim=0) if p_all_acts else torch.empty(0)

    # Build lookup: prefix_id → (acts tensor, meta list)
    prefix_lookup: dict[int, tuple[torch.Tensor, list[dict]]] = {}
    for pid, (pstart, plen) in enumerate(prefix_index):
        prefix_lookup[pid] = (
            prefix_acts_flat[pstart: pstart + plen],
            p_all_meta[pstart: pstart + plen],
        )

    # Load all branch activations into a flat tensor
    b_all_acts, b_all_meta = [], []
    for shard_id in range(num_shards):
        b_all_acts.append(torch.load(
            raw_dir / safe / f"shard_{shard_id:04d}.pt", weights_only=False))
        b_all_meta.extend(torch.load(
            raw_dir / f"meta_shard_{shard_id:04d}.pt", weights_only=False))
    branch_acts_flat = torch.cat(b_all_acts, dim=0) if b_all_acts else torch.empty(0)

    # Reconstruct each entry as [prefix | branch] or [branch] only
    result_acts, result_meta = [], []
    for e in entries:
        b_start, b_len, sample_idx, prefix_id = e[0], e[1], e[2], e[3]
        b_acts = branch_acts_flat[b_start: b_start + b_len]
        b_meta = b_all_meta[b_start: b_start + b_len]

        if include_prefix and prefix_id in prefix_lookup:
            p_acts, p_meta = prefix_lookup[prefix_id]
            result_acts.append(torch.cat([p_acts, b_acts], dim=0))
            result_meta.extend(p_meta + b_meta)
        else:
            result_acts.append(b_acts)
            result_meta.extend(b_meta)

    return (torch.cat(result_acts, dim=0) if result_acts else torch.empty(0),
            result_meta)


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    dataset_tag, dataset_file = resolve_dataset_config(args)
    run_reasoning(dataset_tag, dataset_file)
    