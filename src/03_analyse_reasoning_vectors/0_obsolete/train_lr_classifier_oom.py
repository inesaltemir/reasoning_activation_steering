"""
Logistic Regression Classifier on Raw Activations
===================================================

Trains logistic regression classifiers at three granularity levels
(token, step, sample) using the raw per-token activations stored by
DiskBackedActivationStore (from run_fw_pass.py).

The key insight: when LR is trained on full d_model-dimensional
activation vectors, its learned weight vector **w** ∈ ℝ^d_model is
itself a linear direction in activation space that best separates
correct from incorrect — analogous to (but not identical to) the
contrastive mean-difference "reasoning direction" vector.

Data requirements per granularity:
  Token-level : each token with is_correct ∈ {True, False} is one sample
  Step-level  : tokens grouped by (sample_idx, step_idx), averaged;
                label = True iff all tokens in that step are correct -- look if i can use step labeling?
  Sample-level: tokens grouped by sample_idx, averaged;
                label = per_sample_is_fully_correct from the .pt metadata

Gives OOM error: Process is Killed because we load all shards at once
Total token rows: 2,295,811
Num shards:       23
d_model:          4096
dtype:            torch.bfloat16
Estimated memory:
  Shards on disk (bf16):        18.81 GB
  Loaded as float32:            37.61 GB
  X + sklearn internal copy:    75.23 GB
  StandardScaler doubles it:   ~112.84 GB total
System RAM:
  Total:     168.7 GB
  Available: 98.9 GB
===> need to use streaming
"""

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Train LR classifiers on raw activations at token/step/sample level."
    )
    p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations",
                   help="Path to raw_activations directory produced by run_fw_pass.py")
    p.add_argument("--vectors_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt",
                   help="Path to the reasoning_vectors .pt file (for sample-level labels)")
    p.add_argument("--target_layers", type=int, nargs="+",
                   default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--granularities", type=str, nargs="+",
                   default=["token", "step", "sample"],
                   choices=["token", "step", "sample"])
    p.add_argument("--n_folds", type=int, default=5,
                   help="Number of stratified cross-validation folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Max iterations for LR solver")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverse regularisation strength for LR")
    p.add_argument("--output_dir", type=str, default="results/lr_classifier")
    return p.parse_args()


# ==========================================
# Load activations + metadata from disk shards
# ==========================================
def load_all_shards(raw_dir: Path, hook_name: str, num_shards: int):
    """
    Stream-load all shards for one hook name and return the full
    activation matrix plus aligned metadata.

    Returns
    -------
    activations : torch.Tensor  (total_tokens, d_model) in float32
    metadata    : list[dict]    one dict per token row
    """
    safe = hook_name.replace(".", "_")
    all_acts = []
    all_meta = []

    for shard_id in range(num_shards):
        act_path = raw_dir / safe / f"shard_{shard_id:04d}.pt"
        meta_path = raw_dir / f"meta_shard_{shard_id:04d}.pt"

        acts = torch.load(act_path, weights_only=False).to(torch.float32)
        meta = torch.load(meta_path, weights_only=False)

        all_acts.append(acts)
        all_meta.extend(meta)

    return torch.cat(all_acts, dim=0), all_meta


# ==========================================
# Build (X, y) datasets at each granularity
# ==========================================
def build_token_level(activations: torch.Tensor, metadata: list[dict]):
    """
    Token-level: every token with a definite is_correct label becomes
    one training example.  X = raw activation, y = is_correct.
    """
    mask = []
    labels = []
    for i, m in enumerate(metadata):
        if m["is_correct"] is True:
            mask.append(i)
            labels.append(1)
        elif m["is_correct"] is False:
            mask.append(i)
            labels.append(0)
        # Skip tokens with is_correct == None (prompt / non-reasoning tokens)

    X = activations[mask].numpy()
    y = np.array(labels, dtype=np.int32)
    return X, y


def build_step_level(activations: torch.Tensor, metadata: list[dict]):
    """
    Step-level: group tokens by (sample_idx, step_idx), average their
    activations to get one vector per step.
    
    Label: a step is 'correct' (1) if ALL its tokens have is_correct==True;
    it is 'incorrect' (0) if ANY token has is_correct==False.
    Steps with only None labels are skipped.
    """
    # LOOK INTO THIS: IS THERE NOT A FASTER WAY TO CHECK CORRECTNESS OF STEP? IF LABEL =/= STEP IDX
    # Group token indices by (sample_idx, step_idx)
    step_groups = defaultdict(list)
    for i, m in enumerate(metadata):
        if m["step_idx"] >= 0 and m["is_correct"] is not None:
            key = (m["sample_idx"], m["step_idx"])
            step_groups[key].append(i)

    X_list = []
    y_list = []

    for (sample_idx, step_idx), indices in step_groups.items():
        step_acts = activations[indices]            # (num_tokens_in_step, d_model)
        step_mean = step_acts.mean(dim=0).numpy()   # (d_model,)

        # Label: correct iff every token in the step is correct
        step_labels = [metadata[i]["is_correct"] for i in indices]
        is_correct_step = all(l is True for l in step_labels)

        X_list.append(step_mean)
        y_list.append(1 if is_correct_step else 0)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def build_sample_level(activations: torch.Tensor, metadata: list[dict],
                       per_sample_is_fully_correct: torch.Tensor):
    """
    Sample-level: average ALL reasoning-token activations per sample.
    Label comes from per_sample_is_fully_correct (from the .pt metadata),
    which encodes (label == -1) AND (final_answer_correct == True).
    """
    # Group token indices by sample_idx
    sample_groups = defaultdict(list)
    for i, m in enumerate(metadata):
        sample_groups[m["sample_idx"]].append(i)

    X_list = []
    y_list = []

    for sample_idx in sorted(sample_groups.keys()):
        indices = sample_groups[sample_idx]
        sample_acts = activations[indices]
        sample_mean = sample_acts.mean(dim=0).numpy()

        # Guard against sample_idx exceeding the label tensor
        if sample_idx >= len(per_sample_is_fully_correct):
            continue

        label = int(per_sample_is_fully_correct[sample_idx].item())

        X_list.append(sample_mean)
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ==========================================
# Train & evaluate via cross-validation
# ==========================================
def train_and_evaluate(X, y, n_folds=5, seed=42, C=1.0, max_iter=1000):
    """
    Stratified K-fold cross-validation of a standardised LR pipeline.
    
    Returns a dict of aggregate metrics and the pipeline fitted on all data.
    """
    if len(np.unique(y)) < 2:
        return {"error": "Only one class present — cannot train classifier."}

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",   # handles imbalanced correct/incorrect
            random_state=seed,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc":  "roc_auc",
        "f1":       "f1",
        "precision": "precision",
        "recall":    "recall",
    }

    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
    )

    # Fit on full data for analysis of the learned weight vector
    pipeline.fit(X, y)

    metrics = {}
    for metric_name in scoring:
        test_key = f"test_{metric_name}"
        train_key = f"train_{metric_name}"
        metrics[f"cv_{metric_name}_mean"] = float(np.mean(cv_results[test_key]))
        metrics[f"cv_{metric_name}_std"]  = float(np.std(cv_results[test_key]))
        metrics[f"cv_{metric_name}_train_mean"] = float(np.mean(cv_results[train_key]))

    metrics["n_samples"] = len(y)
    metrics["n_positive"] = int(y.sum())
    metrics["n_negative"] = int(len(y) - y.sum())
    metrics["class_balance"] = float(y.mean())

    return metrics, pipeline, cv_results


# ==========================================
# Analyse the learned LR weight vector
# ==========================================
def analyse_weight_vector(pipeline, reasoning_vectors_layer: dict, layer: int):
    """
    Compare the LR's learned weight vector with the pre-computed
    contrastive reasoning direction vectors.
    
    The LR weight w ∈ ℝ^d_model is the direction the classifier found
    most discriminative.  We measure its cosine similarity with each of
    the stored reasoning_direction_* vectors to see how aligned they are.
    """
    import torch.nn.functional as F

    # Extract weight vector from pipeline (after StandardScaler)
    scaler = pipeline.named_steps["scaler"]
    lr = pipeline.named_steps["lr"]

    # LR operates on scaled features: w_raw = w_scaled / scale
    # (so the direction in original activation space is w / scale)
    w_scaled = lr.coef_[0]                           # (d_model,)
    scale = scaler.scale_                             # (d_model,)
    w_raw = w_scaled / (scale + 1e-12)               # back to original space
    w_raw_tensor = torch.tensor(w_raw, dtype=torch.float32)

    comparisons = {}
    for vec_name, vec in reasoning_vectors_layer.items():
        if "reasoning_direction" in vec_name:
            vec_f32 = vec.to(torch.float32)
            cos_sim = F.cosine_similarity(
                w_raw_tensor.unsqueeze(0),
                vec_f32.unsqueeze(0),
            ).item()
            comparisons[vec_name] = cos_sim

    return comparisons, w_raw


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]

    # Load sample-level labels from the reasoning vectors file
    vectors_data = torch.load(args.vectors_file, weights_only=False)
    per_sample_is_fully_correct = vectors_data["metadata"]["per_sample_is_fully_correct"]

    # Optionally load the reasoning direction vectors for comparison
    layers_dict = vectors_data.get("layers", {})

    all_results = {}

    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'='*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'='*70}")

        # Load all activations for this layer
        print(f"  Loading shards from disk...")
        activations, metadata = load_all_shards(raw_dir, hook_name, num_shards)
        print(f"  Loaded {activations.shape[0]} token rows, shape {activations.shape}")

        layer_results = {}

        for granularity in args.granularities:
            print(f"\n  --- {granularity.upper()} level ---")

            # Build (X, y) for this granularity
            if granularity == "token":
                X, y = build_token_level(activations, metadata)
            elif granularity == "step":
                X, y = build_step_level(activations, metadata)
            elif granularity == "sample":
                X, y = build_sample_level(
                    activations, metadata, per_sample_is_fully_correct
                )

            print(f"  X shape: {X.shape}  |  y: {y.sum()} positive, {len(y)-y.sum()} negative  "
                  f"(balance: {y.mean():.3f})")

            if len(np.unique(y)) < 2:
                print(f"  ⚠ Only one class — skipping.")
                layer_results[granularity] = {"error": "single_class"}
                continue

            # Train & evaluate
            metrics, pipeline, cv_results = train_and_evaluate(
                X, y,
                n_folds=args.n_folds,
                seed=args.seed,
                C=args.C,
                max_iter=args.max_iter,
            )

            print(f"  CV Accuracy : {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
            print(f"  CV AUROC    : {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
            print(f"  CV F1       : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
            print(f"  Train Acc   : {metrics['cv_accuracy_train_mean']:.4f}")

            # Compare LR weight vector with pre-computed reasoning directions
            layer_key = hook_name
            if layer_key in layers_dict:
                comparisons, w_raw = analyse_weight_vector(
                    pipeline, layers_dict[layer_key], layer
                )
                print(f"\n  LR weight vector vs reasoning directions (cosine sim):")
                for vec_name, cos_sim in sorted(comparisons.items()):
                    print(f"    {vec_name:<45} : {cos_sim:+.4f}")
                metrics["weight_vs_reasoning_vectors"] = comparisons

            layer_results[granularity] = metrics

        all_results[str(layer)] = layer_results

        # Free memory before next layer
        del activations, metadata

    # Save results
    results_path = os.path.join(args.output_dir, "lr_classifier_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

# python3 /home/ines/Reasoning-activations/src/03_analyse_reasoning_vectors/train_lr_classifier.py --raw_dir /home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations --vectors_file /home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt --target_layers 18 19 20 21 22 23 24 25 26 27 28 --output_dir results/lr_classifier --C 1.0 --n_folds 5