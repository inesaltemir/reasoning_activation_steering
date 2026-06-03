"""
Evaluate a trained LR classifier on position-balanced .pt eval datasets.

Inputs:
  --lr_dir       : directory containing lr_learned_weights.pt and lr_classifier_results.json
                   (as produced by train_lr_classifier_streaming_extended.py)
  --eval_files   : one or more .pt files with keys {X, y, layer_idx, ...}
                   (as produced by build_position_balanced_eval_dataset.py)
  --granularities: which granularity weight vectors to evaluate (default: all available)

The classifier score is:  score = w_raw · x + intercept_input_space
Prediction:               y_hat = 1 if score > 0 else 0

Usage:
  python eval_lr_on_balanced_dataset.py \
    --lr_dir results/lr_classifier \
    --eval_files balanced_eval_layer22.pt balanced_eval_layer24.pt
"""
import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LR classifier on balanced eval .pt datasets.")
    p.add_argument("--lr_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_no_leak",
                   help="Directory with lr_learned_weights.pt and lr_classifier_results.json")
    
    p.add_argument("--eval_files", type=str, nargs="+",
               default=["/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer18.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer19.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer20.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer21.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer22.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer23.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer24.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer25.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer26.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer27.pt",
                        "/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/balanced_position_eval_data/position_balanced_eval_layer28.pt"],
               help="One or more .pt eval dataset files (with X, y, layer_idx keys)")
    p.add_argument("--granularities", type=str, nargs="+",
               default=["step"],
               help="Granularities to evaluate (default: all available in weights file)")
    p.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_no_leak",
                   help="Optional dir to save JSON results")
    return p.parse_args()


def load_lr_model(lr_dir: Path):
    """Load weight vectors and intercepts from the LR classifier output directory."""
    weights = torch.load(lr_dir / "lr_learned_weights.pt",map_location="cpu", weights_only=False)
    # weights: dict[str(layer_idx)][granularity] -> torch.Tensor (d_model,)

    metrics_path = lr_dir / "lr_classifier_results.json"
    intercepts = {}  # (layer_str, gran) -> float
    if metrics_path.exists():
        with open(metrics_path) as f:
            results_json = json.load(f)
        for layer_str, layer_data in results_json.items():
            if not isinstance(layer_data, dict):
                continue
            for gran, gran_data in layer_data.items():
                if isinstance(gran_data, dict) and "intercept_input_space" in gran_data:
                    intercepts[(layer_str, gran)] = gran_data["intercept_input_space"]

    return weights, intercepts


def evaluate(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray):
    """Compute classifier metrics given weight vector w, intercept b, features X, labels y."""
    scores = X @ w + b
    y_pred = (scores > 0).astype(int)

    try:
        auroc = roc_auc_score(y, scores)
    except ValueError:
        auroc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy":  float(accuracy_score(y, y_pred)),
        "auroc":     float(auroc),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_total": len(y),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "mean_score_pos": float(scores[y == 1].mean()) if (y == 1).any() else None,
        "mean_score_neg": float(scores[y == 0].mean()) if (y == 0).any() else None,
    }


def main():
    args = parse_args()
    lr_dir = Path(args.lr_dir)

    # Load trained model
    weights, intercepts = load_lr_model(lr_dir)
    print(f"Loaded LR weights from: {lr_dir}")
    print(f"  Layers available: {sorted(weights.keys(), key=lambda x: int(x))}")
    print(f"  Intercepts found: {len(intercepts)}")

    all_results = {}

    for eval_path in args.eval_files:
        eval_path = Path(eval_path)
        print(f"\n{'='*70}")
        print(f"  Eval file: {eval_path.name}")
        print(f"{'='*70}")

        data = torch.load(eval_path, weights_only=False)
        X = data["X"].numpy().astype(np.float32)
        y = data["y"].numpy().astype(np.int32)
        layer_idx = data.get("layer_idx", None)
        layer_hook = data.get("layer", None)

        # Infer layer index from hook name if needed
        if layer_idx is None and layer_hook and "blocks." in layer_hook:
            layer_idx = int(layer_hook.split(".")[1])

        layer_str = str(layer_idx)
        print(f"  Layer: {layer_idx}  |  Samples: {len(y)}  "
              f"|  Pos: {y.sum()}  Neg: {len(y) - y.sum()}")

        if layer_str not in weights:
            print(f"  WARNING: layer {layer_str} not found in weights file — skipping.")
            continue

        layer_weights = weights[layer_str]
        granularities = args.granularities or sorted(layer_weights.keys())

        file_results = {}
        for gran in granularities:
            if gran not in layer_weights:
                print(f"  [{gran}] not available — skipping.")
                continue

            w = layer_weights[gran].numpy().astype(np.float64)
            key = (layer_str, gran)

            if key not in intercepts:
                print(f"  [{gran}] no intercept found — skipping.")
                continue

            b = intercepts[key]
            metrics = evaluate(w, b, X.astype(np.float64), y)

            print(f"\n  --- {gran.upper()} ---")
            print(f"    Accuracy  : {metrics['accuracy']:.4f}")
            print(f"    AUROC     : {metrics['auroc']:.4f}")
            print(f"    F1        : {metrics['f1']:.4f}")
            print(f"    Precision : {metrics['precision']:.4f}")
            print(f"    Recall    : {metrics['recall']:.4f}")
            print(f"    Confusion : TP={metrics['tp']}  FP={metrics['fp']}  "
                  f"FN={metrics['fn']}  TN={metrics['tn']}")
            print(f"    Mean score: pos={metrics['mean_score_pos']:.4f}  "
                  f"neg={metrics['mean_score_neg']:.4f}")

            file_results[gran] = metrics

        result_key = f"layer{layer_idx}_{eval_path.stem}"
        all_results[result_key] = file_results

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"position_balanced_eval_layer_results_all_layers.json"
    )

    out = Path(output_file)
    # out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()