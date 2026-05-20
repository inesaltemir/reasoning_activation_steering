"""
Bootstrap Cross-Validation for Reasoning Vector Selection
==========================================================

Takes the output of layer_selection_evaluation.py and performs:
  1. Bootstrap resampling (1000 iterations) to get confidence intervals on AUROC/Cohen's d
  2. Paired comparison between top candidate (vector, layer) pairs
  3. Stability analysis: how often does each candidate "win" across bootstrap samples?
  4. Final recommendation with uncertainty quantification

This addresses a key concern: with small validation sets, the "best" layer may
be unstable. Bootstrap CIs tell you whether the differences are real.

Usage:
  python bootstrap_layer_selection.py \
    --results_file results/layer_selection/layer_selection_results.json \
    --output_dir results/layer_selection \
    --n_bootstrap 1000
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def bootstrap_auroc(pos_scores, neg_scores, n_bootstrap=1000, ci=0.95):
    """Compute AUROC with bootstrap confidence intervals."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    rng = np.random.default_rng(42)
    
    aurocs = []
    for _ in range(n_bootstrap):
        idx_pos = rng.integers(0, n_pos, size=n_pos)
        idx_neg = rng.integers(0, n_neg, size=n_neg)
        
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        scores = np.concatenate([pos_scores[idx_pos], neg_scores[idx_neg]])
        
        try:
            aurocs.append(roc_auc_score(labels, scores))
        except ValueError:
            aurocs.append(0.5)
    
    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    return {
        "mean": np.mean(aurocs),
        "median": np.median(aurocs),
        "ci_lower": np.percentile(aurocs, alpha * 100),
        "ci_upper": np.percentile(aurocs, (1 - alpha) * 100),
        "std": np.std(aurocs),
        "samples": aurocs,
    }


def bootstrap_cohens_d(pos_scores, neg_scores, n_bootstrap=1000, ci=0.95):
    """Compute Cohen's d with bootstrap confidence intervals."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    rng = np.random.default_rng(42)
    
    ds = []
    for _ in range(n_bootstrap):
        idx_pos = rng.integers(0, n_pos, size=n_pos)
        idx_neg = rng.integers(0, n_neg, size=n_neg)
        
        bp = pos_scores[idx_pos]
        bn = neg_scores[idx_neg]
        
        sp = np.std(bp, ddof=1)
        sn = np.std(bn, ddof=1)
        pooled = np.sqrt(((n_pos - 1) * sp**2 + (n_neg - 1) * sn**2) / (n_pos + n_neg - 2))
        
        d = (np.mean(bp) - np.mean(bn)) / (pooled + 1e-10)
        ds.append(d)
    
    ds = np.array(ds)
    alpha = (1 - ci) / 2
    return {
        "mean": np.mean(ds),
        "median": np.median(ds),
        "ci_lower": np.percentile(ds, alpha * 100),
        "ci_upper": np.percentile(ds, (1 - alpha) * 100),
        "std": np.std(ds),
        "samples": ds,
    }


def pairwise_win_rate(auroc_samples_a, auroc_samples_b):
    """How often does A beat B across bootstrap samples?"""
    wins = np.sum(auroc_samples_a > auroc_samples_b)
    return wins / len(auroc_samples_a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/layer_selection")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=5, help="Compare top-k candidates")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results_file) as f:
        results = json.load(f)

    # Identify positive and negative categories
    vec_names = list(results.keys())
    sample_vec = vec_names[0]
    sample_layer = list(results[sample_vec].keys())[0]
    categories = [k for k in results[sample_vec][sample_layer].keys() if not k.startswith("metrics")]
    
    # Determine pos/neg
    pos_cat = None
    for c in categories:
        if "reason" in c.lower() and "non" not in c.lower():
            pos_cat = c
            break
    if pos_cat is None:
        pos_cat = categories[0]
    neg_cats = [c for c in categories if c != pos_cat]
    
    print(f"Positive: {pos_cat}")
    print(f"Negatives: {neg_cats}")
    print(f"Vectors: {vec_names}")
    print(f"Bootstrap samples: {args.n_bootstrap}")

    # Focus on the hardest negative for selection
    hard_neg = [c for c in neg_cats if "hard" in c.lower()]
    eval_neg = hard_neg[0] if hard_neg else neg_cats[0]
    print(f"Primary evaluation negative: {eval_neg}")

    # ==========================================
    # Bootstrap all (vector, layer) pairs
    # ==========================================
    print("\nRunning bootstrap analysis...")
    bootstrap_results = {}
    
    for v_name in vec_names:
        for layer_str in results[v_name]:
            layer_data = results[v_name][layer_str]
            
            pos_scores = np.array(layer_data.get(pos_cat, {}).get("scores", []))
            neg_scores = np.array(layer_data.get(eval_neg, {}).get("scores", []))
            
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue
            
            key = (v_name, int(layer_str))
            auroc_boot = bootstrap_auroc(pos_scores, neg_scores, args.n_bootstrap)
            cohens_boot = bootstrap_cohens_d(pos_scores, neg_scores, args.n_bootstrap)
            
            bootstrap_results[key] = {
                "auroc": auroc_boot,
                "cohens_d": cohens_boot,
            }
    
    # ==========================================
    # Rank candidates
    # ==========================================
    ranked = sorted(
        bootstrap_results.keys(),
        key=lambda k: bootstrap_results[k]["auroc"]["mean"],
        reverse=True,
    )
    
    top_k = ranked[:args.top_k]
    
    print(f"\n{'=' * 90}")
    print(f"  TOP {args.top_k} CANDIDATES (by mean AUROC vs {eval_neg})")
    print(f"{'=' * 90}")
    print(f"  {'Rank':>4} | {'Vector':>40} | {'Layer':>5} | {'AUROC':>20} | {'Cohen d':>20}")
    print(f"  {'─' * 86}")
    
    for i, key in enumerate(top_k):
        v_name, layer = key
        a = bootstrap_results[key]["auroc"]
        d = bootstrap_results[key]["cohens_d"]
        auroc_str = f"{a['mean']:.4f} [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]"
        d_str = f"{d['mean']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
        print(f"  {i+1:>4} | {v_name:>40} | {layer:>5} | {auroc_str:>20} | {d_str:>20}")

    # ==========================================
    # Pairwise comparisons among top candidates
    # ==========================================
    print(f"\n{'=' * 70}")
    print(f"  PAIRWISE WIN RATES (how often row beats column)")
    print(f"{'=' * 70}")
    
    labels = [f"{k[0].split('_')[-1]}@L{k[1]}" for k in top_k]
    win_matrix = np.zeros((len(top_k), len(top_k)))
    
    for i, ki in enumerate(top_k):
        for j, kj in enumerate(top_k):
            if i == j:
                win_matrix[i, j] = 0.5
            else:
                win_matrix[i, j] = pairwise_win_rate(
                    bootstrap_results[ki]["auroc"]["samples"],
                    bootstrap_results[kj]["auroc"]["samples"],
                )
    
    header = "  " + " " * 20 + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"  {label:>20}" + "".join(f"{win_matrix[i, j]:>12.1%}" for j in range(len(top_k)))
        print(row)

    # ==========================================
    # Stability analysis
    # ==========================================
    print(f"\n{'=' * 70}")
    print(f"  STABILITY: How often each candidate is #1 across bootstrap samples")
    print(f"{'=' * 70}")
    
    n_boot = args.n_bootstrap
    win_counts = defaultdict(int)
    
    for b in range(n_boot):
        best_key = None
        best_auroc = -1
        for key in bootstrap_results:
            auroc_b = bootstrap_results[key]["auroc"]["samples"][b]
            if auroc_b > best_auroc:
                best_auroc = auroc_b
                best_key = key
        if best_key:
            win_counts[best_key] += 1
    
    sorted_winners = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_winners[:10]:
        v_name, layer = key
        pct = count / n_boot * 100
        print(f"  {v_name} @ layer {layer}: {pct:.1f}% of bootstrap samples")

    # ==========================================
    # Plot: Bootstrap distributions for top candidates
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_k)))
    
    for i, key in enumerate(top_k):
        v_name, layer = key
        short_label = f"{v_name.replace('reasoning_direction_', '')}@L{layer}"
        
        axes[0].hist(
            bootstrap_results[key]["auroc"]["samples"],
            bins=40, alpha=0.4, color=colors[i], label=short_label,
            edgecolor="none",
        )
        axes[0].axvline(
            bootstrap_results[key]["auroc"]["mean"],
            color=colors[i], linestyle="--", linewidth=1.5,
        )
    
    axes[0].set_xlabel("AUROC", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"Bootstrap AUROC Distributions (vs {eval_neg})", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, key in enumerate(top_k):
        v_name, layer = key
        short_label = f"{v_name.replace('reasoning_direction_', '')}@L{layer}"
        
        axes[1].hist(
            bootstrap_results[key]["cohens_d"]["samples"],
            bins=40, alpha=0.4, color=colors[i], label=short_label,
            edgecolor="none",
        )
        axes[1].axvline(
            bootstrap_results[key]["cohens_d"]["mean"],
            color=colors[i], linestyle="--", linewidth=1.5,
        )
    
    axes[1].set_xlabel("Cohen's d", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(f"Bootstrap Cohen's d Distributions (vs {eval_neg})", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "bootstrap_layer_selection.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved → {plot_path}")

    # ==========================================
    # Final recommendation
    # ==========================================
    best_key = top_k[0]
    best_v, best_l = best_key
    best_a = bootstrap_results[best_key]["auroc"]
    best_d = bootstrap_results[best_key]["cohens_d"]
    
    # Check if the winner is statistically significantly better than #2
    if len(top_k) > 1:
        second_key = top_k[1]
        win_rate = pairwise_win_rate(
            bootstrap_results[best_key]["auroc"]["samples"],
            bootstrap_results[second_key]["auroc"]["samples"],
        )
        sig_str = f"(beats #2 in {win_rate:.0%} of bootstrap samples)"
    else:
        sig_str = ""

    print(f"\n{'=' * 90}")
    print(f"  ★ FINAL RECOMMENDATION")
    print(f"{'=' * 90}")
    print(f"  Vector: {best_v}")
    print(f"  Layer:  {best_l}")
    print(f"  AUROC:  {best_a['mean']:.4f}  95% CI [{best_a['ci_lower']:.4f}, {best_a['ci_upper']:.4f}]")
    print(f"  Cohen:  {best_d['mean']:+.4f}  95% CI [{best_d['ci_lower']:+.4f}, {best_d['ci_upper']:+.4f}]")
    print(f"  {sig_str}")
    
    # Check CI overlap with chance (0.5 AUROC)
    if best_a["ci_lower"] > 0.5:
        print(f"  ✓ AUROC CI is entirely above chance (0.5) — vector has genuine signal")
    else:
        print(f"  ⚠ AUROC CI overlaps with chance (0.5) — vector may not reliably discriminate")
    
    # Check if Cohen's d CI excludes 0
    if best_d["ci_lower"] > 0:
        print(f"  ✓ Cohen's d CI is entirely positive — consistent positive effect")
    else:
        print(f"  ⚠ Cohen's d CI includes 0 — effect direction is unstable")
    
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()