"""
plot_cosine_probe_results.py
============================
Reads cosine_probe_cv_results.json and/or cosine_probe_holdout_results.json
from a given output directory and produces summary plots.

Usage
-----
python plot_cosine_probe_results.py \
    --output_dir /home/ines/Reasoning-activations/results/cosine_probe_eval_layer/processbench

The script auto-discovers:
  - <output_dir>/cosine_probe_cv_results.json          (CV results)
  - <output_dir>/eval_on_<X>/cosine_probe_holdout_results.json   (holdout results)

Metrics plotted
---------------
CV JSON keys per pairing:
  auroc_mean / auroc_std, cohens_d_mean / cohens_d_std,
  balanced_acc_mean / balanced_acc_std, f1_mean / f1_std

Holdout JSON keys per pairing:
  auroc, cohens_d, balanced_acc, f1, mw_p, selectivity
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── colour / style ──────────────────────────────────────────────────────────
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

CV_METRICS = [
    ("auroc_mean",       "auroc_std",       "AUROC"),
    ("cohens_d_mean",    "cohens_d_std",    "Cohen's d"),
    ("balanced_acc_mean","balanced_acc_std", "Balanced Accuracy"),
    ("f1_mean",          "f1_std",          "F1 Score"),
]

HO_METRICS_ALL = [
    ("auroc",        None,  "AUROC"),
    ("cohens_d",     None,  "Cohen's d"),
    ("balanced_acc", None,  "Balanced Accuracy"),
    ("f1",           None,  "F1 Score"),
    ("selectivity",  None,  "Selectivity Ratio"),
    ("mw_p",         None,  "Mann–Whitney p-value"),
]
HO_METRICS = [
    ("auroc",        None,  "AUROC")
]

# ── helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def sorted_layers(results: dict) -> list[int]:
    return sorted(int(k) for k in results.keys())


def extract_cv_series(results: dict, pairing: str, mean_key: str, std_key: str):
    layers = sorted_layers(results)
    means, stds = [], []
    for layer in layers:
        pr = results[str(layer)].get(pairing, {}).get("cv", {})
        means.append(pr.get(mean_key, float("nan")))
        stds.append(pr.get(std_key, float("nan")))
    return layers, np.array(means), np.array(stds)


def extract_ho_series(results: dict, pairing: str, key: str):
    layers = sorted_layers(results)
    vals = []
    for layer in layers:
        pr = results[str(layer)].get(pairing, {}).get("holdout", {})
        vals.append(pr.get(key, float("nan")))
    return layers, np.array(vals)


# ── CV plot ───────────────────────────────────────────────────────────────────
def plot_cv(results: dict, output_dir: Path, origin_dataset: str):
    n_metrics = len(CV_METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    fig.suptitle(
        f"Cosine Probe — Cross-Validation Results\n"
        f"vectors: {origin_dataset}  |  eval: {origin_dataset} (CV folds)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (mean_key, std_key, label) in zip(axes, CV_METRICS):
        for pairing, colour, plabel in zip(PAIRINGS, COLOURS, PAIRING_LABELS):
            layers, means, stds = extract_cv_series(results, pairing, mean_key, std_key)
            ax.plot(layers, means, marker="o", color=colour, label=plabel, linewidth=1.8)
            ax.fill_between(
                layers,
                means - stds,
                means + stds,
                alpha=0.15,
                color=colour,
            )

        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(sorted_layers(results))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Reference line at 0.5 for AUROC / balanced accuracy / F1
        if "auroc" in mean_key or "balanced_acc" in mean_key or "f1" in mean_key:
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        if "cohens_d" in mean_key:
            ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)

    axes[-1].set_xlabel("Layer", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, fontsize=10, frameon=False,
    )

    fig.tight_layout()
    out = output_dir / "cosine_probe_cv_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"CV plot saved → {out}")
    plt.close(fig)


# ── Holdout plot ──────────────────────────────────────────────────────────────
def plot_holdout(results: dict, output_dir: Path, origin_dataset: str, holdout_dataset: str):
    n_metrics = len(HO_METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    fig.suptitle(
        f"Cosine Probe — Hold-out Results\n"
        f"vectors: {origin_dataset}  |  eval: {holdout_dataset}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (key, _, label) in zip(axes, HO_METRICS):
        for pairing, colour, plabel in zip(PAIRINGS, COLOURS, PAIRING_LABELS):
            layers, vals = extract_ho_series(results, pairing, key)
            ax.plot(layers, vals, marker="o", color=colour, label=plabel, linewidth=1.8)

        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(sorted_layers(results))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if key in ("auroc", "balanced_acc", "f1"):
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        if key == "cohens_d":
            ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
        if key == "mw_p":
            ax.set_yscale("log")
            ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0, label="p=0.05")

    axes[-1].set_xlabel("Layer", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, fontsize=10, frameon=False,
    )

    fig.tight_layout()
    out = output_dir / "cosine_probe_holdout_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Holdout plot saved → {out}")
    plt.close(fig)


# ── Combined summary heatmap ──────────────────────────────────────────────────
def plot_summary_heatmap(
    cv_results: dict | None,
    ho_results: dict | None,
    output_dir: Path,
    origin_dataset: str,
    holdout_dataset: str | None,
):
    """
    One heatmap per mode (cv / holdout) showing AUROC across layers × pairings.
    """
    datasets = []
    if cv_results is not None:
        layers = sorted_layers(cv_results)
        mat = np.full((len(PAIRINGS), len(layers)), float("nan"))
        for j, layer in enumerate(layers):
            for i, pairing in enumerate(PAIRINGS):
                pr = cv_results[str(layer)].get(pairing, {}).get("cv", {})
                mat[i, j] = pr.get("auroc_mean", float("nan"))
        datasets.append((
            f"CV — AUROC (mean over folds)\nvectors: {origin_dataset}  |  eval: {origin_dataset} (CV folds)",
            layers, mat,
        ))

    if ho_results is not None:
        layers = sorted_layers(ho_results)
        mat = np.full((len(PAIRINGS), len(layers)), float("nan"))
        for j, layer in enumerate(layers):
            for i, pairing in enumerate(PAIRINGS):
                pr = ho_results[str(layer)].get(pairing, {}).get("holdout", {})
                mat[i, j] = pr.get("auroc", float("nan"))
        ho_label = holdout_dataset if holdout_dataset else "?"
        datasets.append((
            f"Hold-out — AUROC\nvectors: {origin_dataset}  |  eval: {ho_label}",
            layers, mat,
        ))

    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (title, layers, mat) in zip(axes, datasets):
        im = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=9)
        ax.set_yticks(range(len(PAIRINGS)))
        ax.set_yticklabels(PAIRING_LABELS, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=11, fontweight="bold")
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, color="black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="AUROC")

    fig.tight_layout()
    out = output_dir / "cosine_probe_auroc_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved → {out}")
    plt.close(fig)


# ── Combined AUROC across all holdout datasets ───────────────────────────────
def plot_combined_holdout_auroc(
    holdout_entries: list[tuple[str, dict, str]],
    save_dir: Path,
    origin_dataset: str,
):
    """
    One subplot per holdout dataset, all pairings overlaid in each.
    Layout: 1 row × N columns (one column per holdout dataset).
    Saved to <save_dir>/cosine_probe_combined_holdout_auroc.png.
    """
    if not holdout_entries:
        return

    n_cols = len(holdout_entries)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(6 * n_cols, 4),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]  # unwrap the single row

    fig.suptitle(
        f"Cosine Probe — Hold-out AUROC  |  vectors: {origin_dataset}",
        fontsize=13, fontweight="bold",
    )

    for ax, (holdout_dataset, ho_results, _) in zip(axes, holdout_entries):
        layers = sorted_layers(ho_results)
        for pairing, colour, plabel in zip(PAIRINGS, COLOURS, PAIRING_LABELS):
            _, vals = extract_ho_series(ho_results, pairing, "auroc")
            ax.plot(layers, vals, marker="o", color=colour, label=plabel, linewidth=1.8)

        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        ax.set_title(f"eval: {holdout_dataset}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xticks(layers)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.4, 1.02)

    axes[0].set_ylabel("AUROC", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, fontsize=10, frameon=False,
    )

    fig.tight_layout()
    out = save_dir / "cosine_probe_combined_holdout_auroc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Combined holdout AUROC plot saved → {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot cosine probe evaluation results."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ines/Reasoning-activations/results/cosine_probe_eval_layer/processbench",
        help=(
            "Directory containing cosine_probe_cv_results.json and/or "
            "a sub-directory eval_on_<X>/cosine_probe_holdout_results.json."
        ),
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Where to save plots. Defaults to --output_dir.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else base
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Infer dataset names ───────────────────────────────────────────────────
    # Origin dataset: last component of --output_dir
    origin_dataset = base.name

    # ── Load CV results ───────────────────────────────────────────────────────
    cv_path = base / "cosine_probe_cv_results.json"
    cv_results = load_json(cv_path)
    if cv_results:
        print(f"Loaded CV results from {cv_path}")
    else:
        print(f"No CV results found at {cv_path}")

    # ── Load holdout results (search for ALL eval_on_* sub-directories) ─────────
    ho_subdirs = sorted(base.glob("eval_on_*/cosine_probe_holdout_results.json"))
    if ho_subdirs:
        print(f"Found {len(ho_subdirs)} holdout result(s) under {base}/eval_on_*/")
    else:
        print(f"No holdout results found under {base}/eval_on_*/")

    # Collect (holdout_dataset, ho_results) pairs for all subdirectories
    holdout_entries: list[tuple[str, dict]] = []
    for ho_path in ho_subdirs:
        ho_results = load_json(ho_path)
        if ho_results:
            holdout_dataset = ho_path.parent.name.removeprefix("eval_on_")
            holdout_entries.append((holdout_dataset, ho_results, ho_path.parent.name))
            print(f"Loaded holdout results from {ho_path}")
            print(f"  Origin dataset : {origin_dataset}")
            print(f"  Holdout dataset: {holdout_dataset}")

    if cv_results is None and not holdout_entries:
        print("Nothing to plot. Exiting.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    if cv_results:
        plot_cv(cv_results, save_dir, origin_dataset)

    # One set of plots per holdout dataset
    for holdout_dataset, ho_results, subdir_name in holdout_entries:
        holdout_save_dir = save_dir / subdir_name
        holdout_save_dir.mkdir(parents=True, exist_ok=True)
        plot_holdout(ho_results, holdout_save_dir, origin_dataset, holdout_dataset)
        plot_summary_heatmap(cv_results, ho_results, holdout_save_dir, origin_dataset, holdout_dataset)

    # If there were no holdout results, still emit the CV-only heatmap
    if not holdout_entries:
        plot_summary_heatmap(cv_results, None, save_dir, origin_dataset, None)

    # Combined AUROC overview across all holdout datasets
    plot_combined_holdout_auroc(holdout_entries, save_dir, origin_dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()