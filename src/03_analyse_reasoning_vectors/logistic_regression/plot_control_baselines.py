"""
Plot control baselines results from train_lr_classifier_control_baselines.py
Usage:
    python plot_control_baselines.py --input control_baselines_results.json --output_dir plots/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_layer_series(data):
    """Extract per-layer AUROC for each control, assuming step-level granularity."""
    layers = sorted(data.keys(), key=int)
    layers_int = [int(l) for l in layers]

    series = {
        "real":           [],
        "position_only":  [],
        "token_count":    [],
        "residualized":   [],
        "within_problem": [],
        "random_proj":    [],
        "pos_and_length": [],
    }
    # Train AUROCs for overfitting gap
    series["real_train"] = []
    # R² from position
    series["r2_position"] = []

    for l in layers:
        step = data[l].get("step", {})
        series["real"].append(step.get("real_classifier", {}).get("cv_roc_auc_mean", np.nan))
        series["real_train"].append(step.get("real_classifier", {}).get("cv_roc_auc_train_mean", np.nan))
        series["position_only"].append(step.get("ctrl2_position_only", {}).get("cv_roc_auc_mean", np.nan))
        series["token_count"].append(step.get("ctrl3_token_count_only", {}).get("cv_roc_auc_mean", np.nan))
        series["residualized"].append(step.get("ctrl4_position_residualized", {}).get("cv_roc_auc_mean", np.nan))
        series["within_problem"].append(step.get("ctrl5_within_problem", {}).get("cv_roc_auc_mean", np.nan))
        series["pos_and_length"].append(step.get("ctrl7_position_and_length", {}).get("cv_roc_auc_mean", np.nan))

        rp = step.get("ctrl6_random_projection", {})
        series["random_proj"].append(rp.get("random_auroc_mean", np.nan))

        r4 = step.get("ctrl4_position_residualized", {})
        series["r2_position"].append(r4.get("r2_position_explains", np.nan))

    return layers_int, series

def extract_layer_series_sample_granularity(data):
    """Extract per-layer AUROC for each control, assuming sample-level granularity."""
    layers = sorted(data.keys(), key=int)
    layers_int = [int(l) for l in layers]

    series = {
        "real":           [],
        "token_count":    [],
        "random_proj":    [],
    }
    # Train AUROCs for overfitting gap
    series["real_train"] = []

    for l in layers:
        step = data[l].get("sample", {})
        series["real"].append(step.get("real_classifier", {}).get("cv_roc_auc_mean", np.nan))
        series["real_train"].append(step.get("real_classifier", {}).get("cv_roc_auc_train_mean", np.nan))
        #series["position_only"].append(step.get("ctrl2_position_only", {}).get("cv_roc_auc_mean", np.nan))
        series["token_count"].append(step.get("ctrl3_token_count_only", {}).get("cv_roc_auc_mean", np.nan))
        #series["residualized"].append(step.get("ctrl4_position_residualized", {}).get("cv_roc_auc_mean", np.nan))
        #series["within_problem"].append(step.get("ctrl5_within_problem", {}).get("cv_roc_auc_mean", np.nan))
        #series["pos_and_length"].append(step.get("ctrl7_position_and_length", {}).get("cv_roc_auc_mean", np.nan))

        rp = step.get("ctrl6_random_projection", {})
        series["random_proj"].append(rp.get("random_auroc_mean", np.nan))

        #r4 = step.get("ctrl4_position_residualized", {})
        #series["r2_position"].append(r4.get("r2_position_explains", np.nan))

    return layers_int, series


def plot_main_comparison(layers, series, save_dir):
    """Main panel: real classifier vs all controls across layers."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(layers, series["real"], "o-", color="#2563EB", lw=2.5, ms=6,
            label="Real classifier", zorder=5)
    ax.plot(layers, series["residualized"], "s--", color="#7C3AED", lw=2, ms=5,
            label="After regressing out position")
    ax.plot(layers, series["within_problem"], "^--", color="#059669", lw=2, ms=5,
            label="Within-problem centered")
    ax.plot(layers, series["position_only"], "d-.", color="#DC2626", lw=1.8, ms=5,
            label="Position-only")
    ax.plot(layers, series["pos_and_length"], "p-.", color="#EA580C", lw=1.5, ms=5,
            label="Position + length of the step")
    ax.plot(layers, series["token_count"], "x-.", color="#9CA3AF", lw=1.5, ms=6,
            label="Length of the step") # Token-count-only
    ax.plot(layers, series["random_proj"], "v:", color="#6B7280", lw=1.5, ms=5,
            label="Random projection")

    ax.axhline(0.5, color="black", ls=":", lw=1, alpha=0.5, label="Chance")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC (CV)", fontsize=12)
    ax.set_title("Step-level MathShepherd LR classifier vs. control baselines", fontsize=13, fontweight="bold")
    #ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.35))
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_ylim(0.45, 0.92)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "control_baselines_main.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_dir / 'control_baselines_main.png'}")


def plot_main_comparison_sample(layers, series, save_dir):
    """Main panel: real classifier vs all controls across layers."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(layers, series["real"], "o-", color="#2563EB", lw=2.5, ms=6,
            label="Real classifier", zorder=5)
    ax.plot(layers, series["token_count"], "x-.", color="#9CA3AF", lw=1.5, ms=6,
            label="Length of the step") # Token-count-only
    ax.plot(layers, series["random_proj"], "v:", color="#6B7280", lw=1.5, ms=5,
            label="Random projection")

    ax.axhline(0.5, color="black", ls=":", lw=1, alpha=0.5, label="Chance")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC (CV)", fontsize=12)
    ax.set_title("Sample-level MathShepherd LR classifier vs. control baselines", fontsize=13, fontweight="bold") #
    #ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.35))
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_ylim(0.45, 0.92)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "control_baselines_main.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_dir / 'control_baselines_main.png'}")


def plot_signal_decomposition(layers, series, save_dir):
    """Bar chart decomposing how much each confound explains, for one representative layer."""
    # Pick layer with highest real AUROC
    best_idx = int(np.argmax(series["real"]))
    best_layer = layers[best_idx]

    labels = [
        "Real\nclassifier",
        "Position-\nresidualized",
        "Within-\nproblem",
        "Position\nonly",
        "Pos +\nlength",
        "Token-count\nonly",
        "Random\nprojection",
        "Chance",
    ]
    values = [
        series["real"][best_idx],
        series["residualized"][best_idx],
        series["within_problem"][best_idx],
        series["position_only"][best_idx],
        series["pos_and_length"][best_idx],
        series["token_count"][best_idx],
        series["random_proj"][best_idx],
        0.5,
    ]
    colors = ["#2563EB", "#7C3AED", "#059669", "#DC2626", "#EA580C",
              "#9CA3AF", "#6B7280", "black"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", lw=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"Signal Decomposition — Layer {best_layer} (Step Level)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 0.95)
    ax.axhline(0.5, color="black", ls=":", lw=1, alpha=0.4)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "control_baselines_decomposition.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_dir / 'control_baselines_decomposition.png'}")


def plot_overfitting_gap(layers, series, save_dir):
    """Train vs test AUROC to visualize overfitting."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(layers, series["real_train"], "o-", color="#DC2626", lw=2, ms=5,
            label="Train AUROC")
    ax.plot(layers, series["real"], "o-", color="#2563EB", lw=2, ms=5,
            label="Test AUROC (CV)")
    ax.fill_between(layers, series["real"], series["real_train"],
                     color="#DC2626", alpha=0.08)

    # Annotate gap for one layer
    mid = len(layers) // 2
    gap = series["real_train"][mid] - series["real"][mid]
    ax.annotate(f"Gap ≈ {gap:.2f}",
                xy=(layers[mid], (series["real_train"][mid] + series["real"][mid]) / 2),
                fontsize=10, ha="center", color="#DC2626", fontweight="bold")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Train vs. Test AUROC — Overfitting Diagnostic", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0.75, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "control_baselines_overfitting.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_dir / 'control_baselines_overfitting.png'}")


def plot_auroc_delta(layers, series, save_dir):
    """Show the AUROC drop from real classifier when applying each control."""
    real = np.array(series["real"])

    deltas = {
        "Regress out position": real - np.array(series["residualized"]),
        "Within-problem centering": real - np.array(series["within_problem"]),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, delta in deltas.items():
        ax.plot(layers, delta, "o-", lw=2, ms=5, label=label)

    ax.axhline(0, color="black", ls="-", lw=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("ΔAUROC (real − controlled)", fontsize=12)
    ax.set_title("AUROC Drop After Removing Confounds", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "control_baselines_delta.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_dir / 'control_baselines_delta.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_controls_step_granularity/math-shepherd/control_baselines_results.json")
    parser.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/results/lr_classifier_controls_step_granularity/math-shepherd/")
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.input)
    #layers, series = extract_layer_series_sample_granularity(data)
    layers, series = extract_layer_series(data)

    #plot_main_comparison_sample(layers, series, save_dir)
    plot_main_comparison(layers, series, save_dir)
    #plot_signal_decomposition(layers, series, save_dir)
    #plot_overfitting_gap(layers, series, save_dir)
    #plot_auroc_delta(layers, series, save_dir)


if __name__ == "__main__":
    main()