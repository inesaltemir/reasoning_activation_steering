"""
Logistic Regression Classifier on Raw Activations  (Memory-Efficient)
======================================================================

Streams through disk-backed activation shards so that at most ONE shard
(~100k rows × 4096 × float32 ≈ 1.6 GB) is in RAM at a time.

Three granularity levels:
  Token-level  → SGDClassifier(log_loss) with partial_fit, shard by shard
  Step-level   → streaming aggregation into (num_steps, d_model), then LR
  Sample-level → streaming aggregation into (num_samples, d_model), then LR

Usage:
  python train_lr_classifier_streaming.py \
    --raw_dir reasoning_vectors/Qwen3-8B/processbench/raw_activations \
    --vectors_file reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt \
    --target_layers 18 19 20 21 22 23 24 25 26 27 28 \
    --output_dir results/lr_classifier
"""

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings, gc
warnings.filterwarnings("ignore")


# ==========================================
# Args
# ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/raw_activations")
    p.add_argument("--vectors_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_Qwen3-8B_processbench_with_steps_avg_storage.pt")
    p.add_argument("--target_layers", type=int, nargs="+",
                   default=[18])
                   # default=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument("--granularities", type=str, nargs="+",
                   default=["token", "step", "sample"],
                   choices=["token", "step", "sample"])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--sgd_epochs", type=int, default=5,
                   help="Number of full passes over shards for SGD token-level training")
    p.add_argument("--test_shards", type=int, default=4,
                   help="Number of shards held out for token-level testing")
    p.add_argument("--output_dir", type=str, default="results/lr_classifier")
    return p.parse_args()


# ==========================================
# Shard loader (one at a time)
# ==========================================
def load_one_shard(raw_dir: Path, hook_name: str, shard_id: int):
    """Load a single shard + its metadata.  Returns float32 tensor + list[dict]."""
    safe = hook_name.replace(".", "_")
    acts = torch.load(raw_dir / safe / f"shard_{shard_id:04d}.pt",
                      weights_only=False).to(torch.float32)
    meta = torch.load(raw_dir / f"meta_shard_{shard_id:04d}.pt",
                      weights_only=False)
    return acts, meta


# ==========================================
# TOKEN-LEVEL:  SGDClassifier + partial_fit
# ==========================================
def run_token_level(raw_dir, hook_name, num_shards, args):
    """
    Stream through shards.  Last `test_shards` shards are held out.
    Train via SGDClassifier.partial_fit over `sgd_epochs` passes.

    Goal: Predict whether an individual token is correct based on its activation vector.
    Pure streaming. 
    Uses SGDClassifier(loss="log_loss") which mathematically approximates Logistic Regression but supports partial_fit (updating weights batch-by-batch).
    """
    rng = np.random.RandomState(args.seed)

    # Decide train / test split by shard index
    shard_order = list(range(num_shards))
    rng.shuffle(shard_order)
    n_test = min(args.test_shards, max(1, num_shards // 5))
    test_shard_ids  = set(shard_order[:n_test])
    train_shard_ids = [s for s in range(num_shards) if s not in test_shard_ids]

    print(f"    Train shards: {len(train_shard_ids)}  |  Test shards: {n_test}")

    # --- Pass 1: Compute running mean/var for standardisation + class counts ---
    print(f"    Pass 1/{args.sgd_epochs + 2}: computing feature statistics...")
    scaler = StandardScaler()
    class_counts = {0: 0, 1: 0}
    for sid in train_shard_ids:
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
        if mask:
            scaler.partial_fit(acts[mask].numpy())
            for i in mask:
                label = 1 if meta[i]["is_correct"] else 0
                class_counts[label] += 1
        del acts, meta; gc.collect()

    # Compute balanced class weights: w_c = n_total / (n_classes * n_c)
    n_total = class_counts[0] + class_counts[1]
    class_weight_map = {
        c: n_total / (2.0 * count) for c, count in class_counts.items() if count > 0
    }
    print(f"    Class counts: neg={class_counts[0]:,}  pos={class_counts[1]:,}")
    print(f"    Balanced weights: {class_weight_map}")

    # --- Pass 2..N+1: partial_fit SGD ---
    alpha = 1.0 / (args.C * 100_000)   # SGD alpha ≈ 1/(C*n) heuristic
    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=1,           # we control epochs ourselves
        warm_start=True,
        random_state=args.seed,
    )

    classes = np.array([0, 1])
    for epoch in range(args.sgd_epochs):
        epoch_shards = train_shard_ids.copy()
        rng.shuffle(epoch_shards)
        n_seen = 0
        for sid in epoch_shards:
            acts, meta = load_one_shard(raw_dir, hook_name, sid)
            mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
            if not mask:
                del acts, meta; gc.collect()
                continue

            X_shard = scaler.transform(acts[mask].numpy())
            y_shard = np.array([1 if meta[i]["is_correct"] else 0 for i in mask])
            sample_weights = np.array([class_weight_map[yi] for yi in y_shard])
            clf.partial_fit(X_shard, y_shard, classes=classes,
                            sample_weight=sample_weights)
            n_seen += len(mask)
            del acts, meta, X_shard, y_shard, sample_weights; gc.collect()

        print(f"    Epoch {epoch+1}/{args.sgd_epochs}: trained on {n_seen:,} tokens")

    # --- Evaluate on held-out shards ---
    print(f"    Evaluating on test shards...")
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for sid in sorted(test_shard_ids):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        mask = [i for i, m in enumerate(meta) if m["is_correct"] is not None]
        if not mask:
            del acts, meta; gc.collect()
            continue

        X_test = scaler.transform(acts[mask].numpy())
        y_test = np.array([1 if meta[i]["is_correct"] else 0 for i in mask])

        all_y_true.append(y_test)
        all_y_pred.append(clf.predict(X_test))
        all_y_prob.append(clf.decision_function(X_test))
        del acts, meta, X_test, y_test; gc.collect()

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob = np.concatenate(all_y_prob)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auroc":     float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred)),
        "n_test":    int(len(y_true)),
        "n_test_pos": int(y_true.sum()),
        "n_test_neg": int(len(y_true) - y_true.sum()),
        "n_train_shards": len(train_shard_ids),
        "n_test_shards":  n_test,
        "sgd_epochs": args.sgd_epochs,
    }
    return metrics, clf, scaler


# ==========================================
# STEP-LEVEL:  streaming aggregation → LR
# ==========================================
def run_step_level(raw_dir, hook_name, num_shards, d_model, args):
    """
    Stream all shards, accumulate per-step (sample_idx, step_idx) sums
    and token counts.  After all shards: compute means → fit LR.
    Memory: O(num_steps × d_model) — typically a few thousand steps.
    """
    # Accumulate in float64 for numerical stability
    step_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    step_counts = defaultdict(int)
    step_labels = defaultdict(lambda: True)   # AND over is_correct

    print(f"    Streaming {num_shards} shards for step aggregation...")
    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)

        for i, m in enumerate(meta):
            if m["step_idx"] < 0 or m["is_correct"] is None:
                continue
            key = (m["sample_idx"], m["step_idx"])
            step_sums[key]   += acts_np[i]
            step_counts[key] += 1
            if m["is_correct"] is False:
                step_labels[key] = False

        del acts, acts_np, meta; gc.collect()

    # Build X, y
    keys = sorted(step_sums.keys())
    X = np.stack([step_sums[k] / step_counts[k] for k in keys]).astype(np.float32)
    y = np.array([1 if step_labels[k] else 0 for k in keys], dtype=np.int32)

    print(f"    Steps: {len(keys)}  |  Positive: {y.sum()}  |  Negative: {len(y)-y.sum()}")

    # Free the dicts
    del step_sums, step_counts, step_labels; gc.collect()

    return _fit_cv_lr(X, y, args)


# ==========================================
# SAMPLE-LEVEL:  streaming aggregation → LR
# ==========================================
def run_sample_level(raw_dir, hook_name, num_shards, d_model,
                     per_sample_is_fully_correct, args):
    """
    Stream all shards, accumulate per-sample sums.
    Memory: O(num_samples × d_model) — a few hundred samples.
    """
    sample_sums   = defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    sample_counts = defaultdict(int)

    print(f"    Streaming {num_shards} shards for sample aggregation...")
    for sid in range(num_shards):
        acts, meta = load_one_shard(raw_dir, hook_name, sid)
        acts_np = acts.numpy().astype(np.float64)

        for i, m in enumerate(meta):
            sid_sample = m["sample_idx"]
            sample_sums[sid_sample]   += acts_np[i]
            sample_counts[sid_sample] += 1

        del acts, acts_np, meta; gc.collect()

    # Build X, y
    sample_ids = sorted(sample_sums.keys())
    X = np.stack([sample_sums[s] / sample_counts[s] for s in sample_ids]).astype(np.float32)

    y = np.array([
        int(per_sample_is_fully_correct[s].item())
        for s in sample_ids
        if s < len(per_sample_is_fully_correct)
    ], dtype=np.int32)

    # Trim X to match y in case of index mismatch
    X = X[:len(y)]

    print(f"    Samples: {len(y)}  |  Positive: {y.sum()}  |  Negative: {len(y)-y.sum()}")

    del sample_sums, sample_counts; gc.collect()

    return _fit_cv_lr(X, y, args)


# ==========================================
# Shared: CV logistic regression for step/sample
# ==========================================
def _fit_cv_lr(X, y, args):
    """Stratified K-fold CV on a StandardScaler → LR pipeline."""
    if len(np.unique(y)) < 2:
        return {"error": "single_class"}, None

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=args.C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=args.seed)),
    ])

    n_folds = min(args.n_folds, min(np.bincount(y)))  # can't have more folds than minority class
    n_folds = max(n_folds, 2)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                return_train_score=True)

    pipeline.fit(X, y)

    metrics = {}
    for s in scoring:
        metrics[f"cv_{s}_mean"]       = float(np.mean(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_std"]        = float(np.std(cv_results[f"test_{s}"]))
        metrics[f"cv_{s}_train_mean"] = float(np.mean(cv_results[f"train_{s}"]))
    metrics["n_samples"]  = int(len(y))
    metrics["n_positive"] = int(y.sum())
    metrics["n_negative"] = int(len(y) - y.sum())
    metrics["n_folds"]    = n_folds

    return metrics, pipeline


# ==========================================
# Weight vector comparison
# ==========================================
def compare_weights(pipeline_or_clf, scaler, layers_dict, hook_name):
    """Cosine similarity between LR weights and stored reasoning directions."""
    import torch.nn.functional as F

    if isinstance(pipeline_or_clf, Pipeline):
        scaler = pipeline_or_clf.named_steps["scaler"]
        coef = pipeline_or_clf.named_steps["lr"].coef_[0]
    else:
        coef = pipeline_or_clf.coef_[0]

    w_raw = coef / (scaler.scale_ + 1e-12)
    w = torch.tensor(w_raw, dtype=torch.float32)

    if hook_name not in layers_dict:
        return {}

    comps = {}
    for name, vec in layers_dict[hook_name].items():
        if "reasoning_direction" in name and isinstance(vec, torch.Tensor) and vec.dim() == 1:
            cos = F.cosine_similarity(w.unsqueeze(0), vec.float().unsqueeze(0)).item()
            comps[name] = round(cos, 4)
    return comps


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    index = torch.load(raw_dir / "index.pt", weights_only=False)
    num_shards = index["num_shards"]
    d_model    = index["d_model"]

    vectors_data = torch.load(args.vectors_file, weights_only=False)
    per_sample_mask = vectors_data["metadata"]["per_sample_is_fully_correct"]
    layers_dict = vectors_data.get("layers", {})

    all_results = {}
    all_learned_weights = {}

    for layer in args.target_layers:
        hook_name = f"blocks.{layer}.hook_out"
        print(f"\n{'='*70}")
        print(f"  Layer {layer}  ({hook_name})")
        print(f"{'='*70}")

        layer_results = {}

        for gran in args.granularities:
            print(f"\n  --- {gran.upper()} level ---")

            if gran == "token":
                metrics, clf, scaler = run_token_level(raw_dir, hook_name, num_shards, args)
                comps = compare_weights(clf, scaler, layers_dict, hook_name)

            elif gran == "step":
                metrics, pipeline = run_step_level(raw_dir, hook_name, num_shards, d_model, args)
                comps = compare_weights(pipeline, None, layers_dict, hook_name) if pipeline else {}

            elif gran == "sample":
                metrics, pipeline = run_sample_level(
                    raw_dir, hook_name, num_shards, d_model, per_sample_mask, args)
                comps = compare_weights(pipeline, None, layers_dict, hook_name) if pipeline else {}

            if isinstance(metrics, dict) and "error" not in metrics:
                if gran == "token":
                    print(f"    Accuracy : {metrics['accuracy']:.4f}")
                    print(f"    AUROC    : {metrics['auroc']:.4f}")
                    print(f"    F1       : {metrics['f1']:.4f}")
                else:
                    print(f"    CV Accuracy : {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
                    print(f"    CV AUROC    : {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
                    print(f"    CV F1       : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

                if comps:
                    metrics["weight_vs_reasoning_vectors"] = comps
                    print(f"    LR weight vs reasoning directions:")
                    for vn, cs in sorted(comps.items()):
                        print(f"      {vn:<45}: {cs:+.4f}")

            layer_results[gran] = metrics

            # Assuming 'clf' is your trained model (or 'pipeline.named_steps["lr"]' for step/sample)
            # and 'scaler' is your trained scaler.

            if gran == "token":
                w_raw = clf.coef_[0] / (scaler.scale_ + 1e-12)
            else:
                model = pipeline.named_steps["lr"]
                cv_scaler = pipeline.named_steps["scaler"]
                w_raw = model.coef_[0] / (cv_scaler.scale_ + 1e-12)
            # Save it to our dictionary
            if str(layer) not in all_learned_weights:
                all_learned_weights[str(layer)] = {}
            all_learned_weights[str(layer)][gran] = torch.tensor(w_raw, dtype=torch.float32)

            gc.collect()

        all_results[str(layer)] = layer_results

    results_path = os.path.join(args.output_dir, "lr_classifier_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    weights_path = os.path.join(args.output_dir, "lr_learned_weights.pt")
    torch.save(all_learned_weights, weights_path)
    print(f"Weights saved → {weights_path}")


if __name__ == "__main__":
    main()