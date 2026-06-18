#!/usr/bin/env python3
"""
Model Anomaly Detection via Hodge Decomposition.

Key insight: pairwise barrier alone cannot distinguish
  - "far but fixable" (gradient-dominated) — better alignment would help
  - "far and irreparable" (harmonic-dominated) — fundamentally incompatible

For each model, compute:
  - mean barrier (how far from the group)
  - mean harmonic energy fraction (how much of that distance is irreparable)

Then validate:
  - "fixable outliers" (high barrier, low harmonic) should be fine within same-seed group
  - "toxic outliers" (high barrier, high harmonic) should be bad against everyone
"""

import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.topology.hodge import (
    build_mergeability_complex, build_boundary_operators,
    hodge_decomposition,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    # Load 24-model data
    barrier_path = os.path.join(ROOT, "results_24", "barriers_plain.npz")
    data = np.load(barrier_path, allow_pickle=True)
    barrier_matrix = data["barrier_matrix"]
    stored_names = list(data["model_names"])
    triplet_keys, triplet_vals = data["triplet_keys"], data["triplet_vals"]
    triplet_barriers = {tuple(sorted(k)): float(v) for k, v in zip(triplet_keys, triplet_vals)}

    n = barrier_matrix.shape[0]
    short_names = [sn.replace("plaincnn_", "").replace("_wd0.0001_augbasic.pt", "") for sn in stored_names]

    # Parse metadata
    seeds = np.array([int(s.split("_")[0].replace("s", "")) for s in short_names])
    lrs = np.array([float(s.split("_")[1].replace("lr", "")) for s in short_names])

    # Load test accuracies
    soup_path = os.path.join(ROOT, "results", "soup_experiment.json")
    with open(soup_path) as f:
        soup_data = json.load(f)
    test_accs = np.array([soup_data["individual_test_accs"].get(sn, 0) for sn in short_names])

    # Build complex at median tau
    upper = barrier_matrix[np.triu_indices(n, k=1)]
    tau = np.median(upper)
    V, E, T = build_mergeability_complex(barrier_matrix, triplet_barriers, tau)
    B1, B2 = build_boundary_operators(V, E, T)
    edge_to_idx = {e: idx for idx, e in enumerate(E)}

    # Hodge decomposition
    f = np.array([barrier_matrix[e[0], e[1]] for e in E])
    decomp = hodge_decomposition(B1, B2, f)

    grad_sq = decomp['gradient']**2
    curl_sq = decomp['curl']**2
    harm_sq = decomp['harmonic']**2

    # =========================================================================
    # Per-model aggregation
    # =========================================================================
    model_stats = []
    for i in range(n):
        # All edges touching model i
        edges_i = [(idx, e) for idx, e in enumerate(E) if i in e]

        if not edges_i:
            model_stats.append({
                'name': short_names[i], 'seed': seeds[i], 'lr': lrs[i],
                'test_acc': test_accs[i],
                'mean_barrier': barrier_matrix[i].sum() / (n - 1),
                'n_edges_in_complex': 0,
                'mean_harm_frac': 0, 'mean_grad_frac': 0, 'mean_curl_frac': 0,
                'total_harm_energy': 0, 'total_grad_energy': 0,
            })
            continue

        edge_indices = [idx for idx, e in edges_i]
        total_per_edge = grad_sq[edge_indices] + curl_sq[edge_indices] + harm_sq[edge_indices]
        total_per_edge = np.maximum(total_per_edge, 1e-12)

        harm_fracs = harm_sq[edge_indices] / total_per_edge
        grad_fracs = grad_sq[edge_indices] / total_per_edge
        curl_fracs = curl_sq[edge_indices] / total_per_edge

        model_stats.append({
            'name': short_names[i],
            'seed': int(seeds[i]),
            'lr': float(lrs[i]),
            'test_acc': float(test_accs[i]),
            'mean_barrier': float(barrier_matrix[i].sum() / (n - 1)),
            'n_edges_in_complex': len(edges_i),
            'mean_harm_frac': float(np.mean(harm_fracs)),
            'mean_grad_frac': float(np.mean(grad_fracs)),
            'mean_curl_frac': float(np.mean(curl_fracs)),
            'total_harm_energy': float(np.sum(harm_sq[edge_indices])),
            'total_grad_energy': float(np.sum(grad_sq[edge_indices])),
        })

    # =========================================================================
    # Print table sorted by harmonic fraction
    # =========================================================================
    print(f"{'='*90}")
    print(f"MODEL ANOMALY PROFILE (τ={tau:.4f}, {len(E)} edges, {len(T)} triangles)")
    print(f"{'='*90}")
    print(f"  {'Model':>15s}  {'seed':>4s}  {'LR':>5s}  {'acc%':>5s}  "
          f"{'barrier':>8s}  {'grad%':>6s}  {'curl%':>6s}  {'harm%':>6s}  {'#edges':>6s}")
    print("-" * 90)

    sorted_stats = sorted(model_stats, key=lambda x: -x['mean_harm_frac'])
    for m in sorted_stats:
        print(f"  {m['name']:>15s}  {m['seed']:4d}  {m['lr']:5.3f}  {m['test_acc']:5.1f}  "
              f"{m['mean_barrier']:8.4f}  {100*m['mean_grad_frac']:6.1f}  "
              f"{100*m['mean_curl_frac']:6.1f}  {100*m['mean_harm_frac']:6.1f}  "
              f"{m['n_edges_in_complex']:6d}")

    # =========================================================================
    # Classify models: fixable vs toxic
    # =========================================================================
    barrier_vals = np.array([m['mean_barrier'] for m in model_stats])
    harm_vals = np.array([m['mean_harm_frac'] for m in model_stats])

    barrier_median = np.median(barrier_vals)
    harm_median = np.median(harm_vals)

    print(f"\nMedian barrier: {barrier_median:.4f}")
    print(f"Median harmonic fraction: {100*harm_median:.1f}%")

    print(f"\n{'='*70}")
    print("QUADRANT ANALYSIS")
    print(f"{'='*70}")

    quadrants = {
        'Compatible (low barrier, low harm)': [],
        'Fixable outlier (high barrier, low harm)': [],
        'Toxic outlier (high barrier, high harm)': [],
        'Surprising (low barrier, high harm)': [],
    }
    for m in model_stats:
        hi_b = m['mean_barrier'] > barrier_median
        hi_h = m['mean_harm_frac'] > harm_median
        if not hi_b and not hi_h:
            quadrants['Compatible (low barrier, low harm)'].append(m)
        elif hi_b and not hi_h:
            quadrants['Fixable outlier (high barrier, low harm)'].append(m)
        elif hi_b and hi_h:
            quadrants['Toxic outlier (high barrier, high harm)'].append(m)
        else:
            quadrants['Surprising (low barrier, high harm)'].append(m)

    for label, models in quadrants.items():
        if not models:
            print(f"\n  {label}: (none)")
            continue
        names = [m['name'] for m in models]
        s = [m['seed'] for m in models]
        print(f"\n  {label} ({len(models)} models):")
        print(f"    Models: {names}")
        print(f"    Seeds: {s}")
        print(f"    Mean barrier: {np.mean([m['mean_barrier'] for m in models]):.4f}")
        print(f"    Mean harm%:   {100*np.mean([m['mean_harm_frac'] for m in models]):.1f}%")
        print(f"    Mean acc:     {np.mean([m['test_acc'] for m in models]):.2f}%")

    # =========================================================================
    # Validation: same-seed barrier for fixable vs toxic
    # =========================================================================
    print(f"\n{'='*70}")
    print("VALIDATION: Within-seed vs Cross-seed Barriers")
    print(f"{'='*70}")

    for m in model_stats:
        same_seed = [j for j in range(n) if seeds[j] == m['seed'] and j != short_names.index(m['name'])]
        diff_seed = [j for j in range(n) if seeds[j] != m['seed']]
        i = short_names.index(m['name'])

        if same_seed:
            within = np.mean([barrier_matrix[i, j] for j in same_seed])
        else:
            within = 0
        cross = np.mean([barrier_matrix[i, j] for j in diff_seed])
        m['within_seed_barrier'] = float(within)
        m['cross_seed_barrier'] = float(cross)

    print(f"\n  {'Model':>15s}  {'harm%':>6s}  {'within_seed':>12s}  {'cross_seed':>11s}  {'ratio':>6s}")
    print("-" * 65)
    for m in sorted(model_stats, key=lambda x: -x['mean_harm_frac']):
        ratio = m['cross_seed_barrier'] / max(m['within_seed_barrier'], 1e-8)
        print(f"  {m['name']:>15s}  {100*m['mean_harm_frac']:6.1f}  "
              f"{m['within_seed_barrier']:12.4f}  {m['cross_seed_barrier']:11.4f}  {ratio:6.1f}x")

    # =========================================================================
    # Key test: does harmonic fraction predict "unfixable"?
    # =========================================================================
    print(f"\n{'='*70}")
    print("KEY TEST: Harmonic fraction as 'unfixability' signal")
    print(f"{'='*70}")

    # For each model: within-seed barrier measures "fixable distance"
    # If harmonic is high, even within-seed barrier should be high (can't fix)
    # If harmonic is low, within-seed barrier should be low (fixable by alignment)
    within_barriers = np.array([m['within_seed_barrier'] for m in model_stats])
    cross_barriers = np.array([m['cross_seed_barrier'] for m in model_stats])
    barrier_reduction = (cross_barriers - within_barriers) / np.maximum(cross_barriers, 1e-8)

    r_harm_within, p_harm_within = stats.spearmanr(harm_vals, within_barriers)
    r_harm_cross, p_harm_cross = stats.spearmanr(harm_vals, cross_barriers)
    r_harm_reduction, p_harm_reduction = stats.spearmanr(harm_vals, barrier_reduction)

    print(f"  harm_frac vs within-seed barrier: ρ={r_harm_within:+.3f} (p={p_harm_within:.4f})")
    print(f"  harm_frac vs cross-seed barrier:  ρ={r_harm_cross:+.3f} (p={p_harm_cross:.4f})")
    print(f"  harm_frac vs barrier reduction:   ρ={r_harm_reduction:+.3f} (p={p_harm_reduction:.4f})")
    print(f"    (barrier reduction = how much within-seed helps vs cross-seed)")

    # Per-seed: average harm fraction
    print(f"\n  Per-seed average harmonic fraction:")
    for s in sorted(set(seeds)):
        mask = seeds == s
        h = np.mean(harm_vals[mask])
        b = np.mean(barrier_vals[mask])
        print(f"    Seed {s}: harm={100*h:.1f}%, barrier={b:.4f}")

    # =========================================================================
    # Scatter plot data for paper figure
    # =========================================================================
    print(f"\n{'='*70}")
    print("SCATTER PLOT DATA (x=mean_barrier, y=harm_frac, color=seed)")
    print(f"{'='*70}")
    print(f"  {'name':>15s}  {'x_barrier':>10s}  {'y_harm':>8s}  {'seed':>4s}  {'lr':>5s}")
    for m in model_stats:
        print(f"  {m['name']:>15s}  {m['mean_barrier']:10.4f}  "
              f"{m['mean_harm_frac']:8.4f}  {m['seed']:4d}  {m['lr']:5.3f}")

    # Generate the scatter plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_seeds = sorted(set(seeds))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_seeds)))
    seed_to_color = {s: colors[i] for i, s in enumerate(unique_seeds)}

    for m in model_stats:
        c = seed_to_color[m['seed']]
        marker = {'0.01': 'o', '0.05': 's', '0.1': '^'}.get(f"{m['lr']}", 'D')
        ax.scatter(m['mean_barrier'], m['mean_harm_frac'],
                   c=[c], marker=marker, s=60, edgecolors='black', linewidth=0.5,
                   zorder=3)

    # Quadrant lines
    ax.axvline(x=barrier_median, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=harm_median, color='gray', linestyle='--', alpha=0.5)

    # Labels
    ax.set_xlabel('Mean Pairwise Barrier', fontsize=12)
    ax.set_ylabel('Mean Harmonic Fraction', fontsize=12)
    ax.set_title('Model Anomaly Profile: Barrier vs Harmonic Fraction', fontsize=13)

    # Legend for seeds
    for s in unique_seeds:
        ax.scatter([], [], c=[seed_to_color[s]], label=f'seed {s}', s=40)
    # Legend for LR
    for lr, marker in [('0.01', 'o'), ('0.05', 's'), ('0.1', '^')]:
        ax.scatter([], [], c='gray', marker=marker, label=f'LR={lr}', s=40)
    ax.legend(fontsize=8, ncol=2, loc='upper left')

    # Quadrant labels
    ax.text(0.02, 0.02, 'Compatible', transform=ax.transAxes, fontsize=9, color='green', alpha=0.7)
    ax.text(0.75, 0.02, 'Fixable\noutlier', transform=ax.transAxes, fontsize=9, color='orange', alpha=0.7)
    ax.text(0.75, 0.92, 'Toxic\noutlier', transform=ax.transAxes, fontsize=9, color='red', alpha=0.7)

    plt.tight_layout()
    fig_path = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, "anomaly_scatter.png"), dpi=150)
    plt.close()
    print(f"\nFigure saved to results/figures/anomaly_scatter.png")

    # Save results
    out = {
        "tau": float(tau),
        "model_stats": model_stats,
        "quadrants": {k: [m['name'] for m in v] for k, v in quadrants.items()},
        "correlations": {
            "harm_vs_within_seed": {"rho": float(r_harm_within), "p": float(p_harm_within)},
            "harm_vs_cross_seed": {"rho": float(r_harm_cross), "p": float(p_harm_cross)},
            "harm_vs_reduction": {"rho": float(r_harm_reduction), "p": float(p_harm_reduction)},
        },
    }
    out_path = os.path.join(ROOT, "results", "anomaly_detection.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
