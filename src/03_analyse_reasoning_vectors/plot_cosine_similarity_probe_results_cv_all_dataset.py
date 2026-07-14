"""
plot_cv_auroc_all_datasets.py
==============================
Reads cosine_probe_cv_results.json from every sub-directory of
  /home/ines/Reasoning-activations/results/cosine_probe_eval_layer/
and plots CV-fold AUROC (mean ± std) for all pairings in a single figure,
one subplot per dataset.

Usage
-----
python plot_cv_auroc_all_datasets.py \
    [--base_dir /home/ines/Reasoning-activations/results/cosine_probe_eval_layer] \
    [--save_path ./cv_auroc_all_datasets.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIRINGS = [
    "step_dir→step_eval",
    "step_dir→sample_eval",
    "sample_dir→step_eval",
    "sample_dir→sample_eval",
]
PAIRING_LABELS = [
    "step dir → step eval",
    "step dir → sample eval",
    "sample dir → step eval",
    "sample dir → sample eval",
]
COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def sorted_layers(results: dict) -> list[int]:
    return sorted(int(k) for k in results.keys())


def extract_cv_auroc(results: dict, pairing: str):
    layers = sorted_layers(results)
    means, stds = [], []
    for layer in layers:
        pr = results[str(layer)].get(pairing, {}).get("cv", {})
        means.append(pr.get("auroc_mean", float("nan")))
        stds.append(pr.get("auroc_std", float("nan")))
    return layers, np.array(means), np.array(stds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_dir",
        default="/home/ines/Reasoning-activations/results/cosine_probe_eval_layer",
    )
    p.add_argument("--save_path", default="./cv_auroc_all_datasets.png")
    args = p.parse_args()

    base = Path(args.base_dir)

    # Collect all datasets that have a CV results file
    entries = []
    for cv_path in sorted(base.glob("*/cosine_probe_cv_results.json")):
        results = load_json(cv_path)
        if results:
            entries.append((cv_path.parent.name, results))

    if not entries:
        print(f"No cosine_probe_cv_results.json files found under {base}/*/")
        return

    print(f"Found {len(entries)} dataset(s): {[e[0] for e in entries]}")

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True, squeeze=False)
    axes = axes[0]

    fig.suptitle("Cosine Probe — CV-fold AUROC", fontsize=14, fontweight="bold")

    for ax, (dataset_name, results) in zip(axes, entries):
        layers = sorted_layers(results)
        for pairing, colour, label in zip(PAIRINGS, COLOURS, PAIRING_LABELS):
            lyrs, means, stds = extract_cv_auroc(results, pairing)
            ax.plot(lyrs, means, marker="o", color=colour, label=label, linewidth=1.8)
            ax.fill_between(lyrs, means - stds, means + stds, alpha=0.15, color=colour)

        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xticks(layers)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.4, 1.02)

    axes[0].set_ylabel("AUROC (CV mean ± std)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, fontsize=10, frameon=False,
    )

    fig.tight_layout()
    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()