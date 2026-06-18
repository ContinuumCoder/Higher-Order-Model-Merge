#!/usr/bin/env python3
"""
Hodge Attribution Analysis: decompose merge barriers into gradient, curl, harmonic.

Key insight: the Hodge decomposition of barrier values on the mergeability complex
gives three fundamentally different types of incompatibility:

  - Gradient: barrier ~ potential difference. Models have a scalar "merge potential" s.
    Compatible models have similar s. This is a new signal, distinct from accuracy.
  - Curl: directional asymmetry. A->B->C->A forms a cycle with nonzero circulation.
    Merge order matters (relevant for task arithmetic, not just averaging).
  - Harmonic: irreducible topological obstruction. No node potential or local patch
    can explain it. These edges are "fundamentally incompatible".

Uses existing 24-model barrier data — no new computation needed.
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
    hodge_analysis, hodge_decomposition,
)


ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_node_potential(B1, f_grad):
    """Extract node potential s such that f_grad ≈ B1^T s.

    Solves: s = argmin ||B1^T s - f_grad||^2
    i.e. B1 B1^T s = B1 f_grad
    """
    B1_dense = B1.toarray().astype(np.float64)
    L0 = B1_dense @ B1_dense.T
    rhs = B1_dense @ f_grad
    # Use pseudoinverse (L0 is singular: constant vector in kernel)
    s = np.linalg.pinv(L0) @ rhs
    # Normalize: zero-mean
    s = s - s.mean()
    return s


def main():
    # Load 24-model barrier data
    barrier_path = os.path.join(ROOT, "results_24", "barriers_plain.npz")
    if not os.path.exists(barrier_path):
        barrier_path = os.path.join(ROOT, "results", "barriers_plain.npz")
    print(f"Loading barriers from {barrier_path}")

    data = np.load(barrier_path, allow_pickle=True)
    barrier_matrix = data["barrier_matrix"]
    triplet_keys = data["triplet_keys"]
    triplet_vals = data["triplet_vals"]
    stored_names = list(data["model_names"])

    triplet_barriers = {}
    for key, val in zip(triplet_keys, triplet_vals):
        triplet_barriers[tuple(sorted(key))] = float(val)

    n = barrier_matrix.shape[0]
    upper = barrier_matrix[np.triu_indices(n, k=1)]

    # Parse model metadata from names
    short_names = []
    lrs = []
    seeds = []
    for sn in stored_names:
        short = sn.replace("plaincnn_", "").replace("_wd0.0001_augbasic.pt", "")
        short_names.append(short)
        parts = short.split("_")
        seeds.append(int(parts[0].replace("s", "")))
        lrs.append(float(parts[1].replace("lr", "")))

    lrs = np.array(lrs)
    seeds = np.array(seeds)

    # Load individual test accuracies from report
    report_path = os.path.join(ROOT, "results_24", "report.json")
    if not os.path.exists(report_path):
        report_path = os.path.join(ROOT, "results", "report.json")

    # Use the soup experiment data for individual accuracies
    soup_path = os.path.join(ROOT, "results", "soup_experiment.json")
    if os.path.exists(soup_path):
        with open(soup_path) as f:
            soup_data = json.load(f)
        test_accs = np.array([soup_data["individual_test_accs"].get(sn, 0)
                              for sn in short_names])
        val_accs = np.array([soup_data["individual_val_accs"].get(sn, 0)
                             for sn in short_names])
    else:
        test_accs = np.zeros(n)
        val_accs = np.zeros(n)

    # =========================================================================
    # Hodge decomposition at multiple tau values
    # =========================================================================
    # Pick tau values where the complex is interesting (has edges and unfilled triangles)
    tau_values = np.percentile(upper, [25, 50, 60, 70, 75, 80, 85])

    print(f"\n{'='*70}")
    print(f"HODGE ATTRIBUTION ANALYSIS ({n} models)")
    print(f"{'='*70}")

    results = {}

    for tau in tau_values:
        V, E, T = build_mergeability_complex(barrier_matrix, triplet_barriers, tau)
        if len(E) < 3:
            continue

        B1, B2 = build_boundary_operators(V, E, T)
        ha = hodge_analysis(B1, B2)

        # 1-cochain: barrier values on edges
        f = np.array([barrier_matrix[e[0], e[1]] for e in E])
        decomp = hodge_decomposition(B1, B2, f)

        total_energy = max(decomp['norm_f']**2, 1e-12)
        grad_pct = 100 * decomp['norm_gradient']**2 / total_energy
        curl_pct = 100 * decomp['norm_curl']**2 / total_energy
        harm_pct = 100 * decomp['norm_harmonic']**2 / total_energy

        print(f"\n--- τ = {tau:.4f} ({len(E)} edges, {len(T)} triangles, "
              f"β₀={ha['beta_0']}, β₁={ha['beta_1']}) ---")
        print(f"  Energy: gradient {grad_pct:.1f}% | curl {curl_pct:.1f}% | harmonic {harm_pct:.1f}%")

        if grad_pct < 1 and harm_pct < 1:
            continue

        # ==================================================================
        # 1. NODE POTENTIAL from gradient component
        # ==================================================================
        s = extract_node_potential(B1, decomp['gradient'])

        print(f"\n  [1] Node Potential s (merge compatibility potential):")
        print(f"      {'Model':>15s}  {'s':>8s}  {'LR':>6s}  {'Seed':>4s}  {'Test%':>7s}")
        order = np.argsort(s)
        for idx in order:
            print(f"      {short_names[idx]:>15s}  {s[idx]:8.4f}  {lrs[idx]:6.3f}  "
                  f"{seeds[idx]:4d}  {test_accs[idx]:7.2f}")

        # Correlations
        r_lr, p_lr = stats.spearmanr(s, lrs)
        r_acc, p_acc = stats.spearmanr(s, test_accs) if test_accs.any() else (0, 1)
        r_seed, p_seed = stats.spearmanr(s, seeds)

        # LR groups: average s within each LR
        unique_lrs = sorted(set(lrs))
        lr_group_s = {lr: s[lrs == lr].mean() for lr in unique_lrs}

        print(f"\n      Spearman correlations:")
        print(f"        s vs LR:       ρ={r_lr:+.3f}  (p={p_lr:.4f})")
        print(f"        s vs test_acc: ρ={r_acc:+.3f}  (p={p_acc:.4f})")
        print(f"        s vs seed:     ρ={r_seed:+.3f}  (p={p_seed:.4f})")
        print(f"      Mean s by LR group: {', '.join(f'lr={lr}: {v:.4f}' for lr, v in lr_group_s.items())}")

        # Check if s captures something beyond LR and accuracy
        # Partial correlation: s vs acc controlling for LR
        if len(unique_lrs) > 1 and test_accs.any():
            from sklearn.linear_model import LinearRegression
            lr_resid_s = s - LinearRegression().fit(lrs.reshape(-1, 1), s).predict(lrs.reshape(-1, 1))
            lr_resid_acc = test_accs - LinearRegression().fit(lrs.reshape(-1, 1), test_accs).predict(lrs.reshape(-1, 1))
            r_partial, p_partial = stats.spearmanr(lr_resid_s, lr_resid_acc)
            print(f"        s vs acc (partial, controlling LR): ρ={r_partial:+.3f}  (p={p_partial:.4f})")

        # ==================================================================
        # 2. HARMONIC EDGES — irreducible incompatibility
        # ==================================================================
        harm_vec = decomp['harmonic']
        harm_energy_per_edge = harm_vec**2

        print(f"\n  [2] Harmonic Edges (irreducible incompatibility):")
        if harm_energy_per_edge.sum() > 1e-12:
            top_harm = np.argsort(harm_energy_per_edge)[::-1][:10]
            print(f"      {'Edge':>25s}  {'|h|²':>10s}  {'barrier':>8s}  {'%total':>7s}")
            for idx in top_harm:
                i, j = E[idx]
                pct = 100 * harm_energy_per_edge[idx] / max(harm_energy_per_edge.sum(), 1e-12)
                print(f"      {short_names[i]:>12s}-{short_names[j]:<12s}  "
                      f"{harm_energy_per_edge[idx]:10.6f}  {barrier_matrix[i,j]:8.4f}  {pct:6.1f}%")

            # Are harmonic-heavy edges between specific LR groups?
            harm_by_lr_pair = {}
            for idx in range(len(E)):
                i, j = E[idx]
                lr_pair = tuple(sorted([lrs[i], lrs[j]]))
                if lr_pair not in harm_by_lr_pair:
                    harm_by_lr_pair[lr_pair] = []
                harm_by_lr_pair[lr_pair].append(harm_energy_per_edge[idx])
            print(f"\n      Harmonic energy by LR pair:")
            for lr_pair in sorted(harm_by_lr_pair):
                vals = harm_by_lr_pair[lr_pair]
                print(f"        LR {lr_pair}: mean |h|² = {np.mean(vals):.6f} ({len(vals)} edges)")
        else:
            print(f"      No harmonic energy (fully gradient+curl)")

        # ==================================================================
        # 3. CURL TRIANGLES — directional asymmetry
        # ==================================================================
        curl_vec = decomp['curl']

        print(f"\n  [3] Curl Analysis (directional asymmetry):")
        if len(T) > 0 and np.linalg.norm(curl_vec) > 1e-8:
            # curl = B2 c, so c = pinv(B2^T B2) B2^T curl_vec
            B2_dense = B2.toarray().astype(np.float64)
            if B2_dense.shape[1] > 0:
                c = np.linalg.pinv(B2_dense.T @ B2_dense) @ B2_dense.T @ curl_vec
                curl_per_tri = c**2
                top_curl = np.argsort(curl_per_tri)[::-1][:10]
                print(f"      {'Triangle':>40s}  {'|c|²':>10s}  {'max_pw':>8s}  {'triplet':>8s}")
                for idx in top_curl:
                    i, j, k = T[idx]
                    pw_max = max(barrier_matrix[i,j], barrier_matrix[i,k], barrier_matrix[j,k])
                    tb = triplet_barriers.get(tuple(sorted([i,j,k])), -1)
                    print(f"      {short_names[i]:>12s}-{short_names[j]}-{short_names[k]:<12s}  "
                          f"{curl_per_tri[idx]:10.6f}  {pw_max:8.4f}  {tb:8.4f}")
        else:
            print(f"      No curl (no triangles or zero curl component)")

        # ==================================================================
        # 4. POTENTIAL-GUIDED SOUP: merge s-nearby models
        # ==================================================================
        print(f"\n  [4] Potential-Guided Grouping:")
        # Cluster models by s into groups of nearby potential
        order = np.argsort(s)
        # Sliding window: groups of 3-5 consecutive models in s-order
        for group_size in [3, 5]:
            best_group = None
            best_mean_barrier = float('inf')
            for start in range(len(order) - group_size + 1):
                group = order[start:start + group_size]
                # Mean pairwise barrier within group
                barriers = [barrier_matrix[group[a], group[b]]
                            for a, b in combinations(range(group_size), 2)]
                mean_b = np.mean(barriers)
                if mean_b < best_mean_barrier:
                    best_mean_barrier = mean_b
                    best_group = group

            if best_group is not None:
                group_names = [short_names[i] for i in best_group]
                group_s = [s[i] for i in best_group]
                group_acc = [test_accs[i] for i in best_group]
                barriers_in_group = [barrier_matrix[best_group[a], best_group[b]]
                                     for a, b in combinations(range(len(best_group)), 2)]
                print(f"      Best {group_size}-model group by s-proximity:")
                print(f"        Models: {group_names}")
                print(f"        s values: [{min(group_s):.4f}, {max(group_s):.4f}] "
                      f"(spread: {max(group_s)-min(group_s):.4f})")
                print(f"        Test acc: {[f'{a:.1f}' for a in group_acc]}")
                print(f"        Pairwise barriers: mean={np.mean(barriers_in_group):.4f} "
                      f"max={np.max(barriers_in_group):.4f}")

        # Store results for this tau
        results[f"tau_{tau:.4f}"] = {
            "tau": float(tau),
            "n_edges": len(E), "n_triangles": len(T),
            "beta_0": ha["beta_0"], "beta_1": ha["beta_1"],
            "energy_gradient_pct": grad_pct,
            "energy_curl_pct": curl_pct,
            "energy_harmonic_pct": harm_pct,
            "node_potential": {short_names[i]: float(s[i]) for i in range(n)},
            "correlation_s_lr": {"rho": float(r_lr), "p": float(p_lr)},
            "correlation_s_acc": {"rho": float(r_acc), "p": float(p_acc)},
            "correlation_s_seed": {"rho": float(r_seed), "p": float(p_seed)},
        }

    # =========================================================================
    # Summary across tau values
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Energy Decomposition Across Scales")
    print(f"{'='*70}")
    print(f"  {'tau':>8s}  {'edges':>6s}  {'tri':>5s}  {'β₁':>3s}  "
          f"{'grad%':>6s}  {'curl%':>6s}  {'harm%':>6s}  {'ρ(s,LR)':>8s}  {'ρ(s,acc)':>9s}")
    for key in sorted(results):
        r = results[key]
        print(f"  {r['tau']:8.4f}  {r['n_edges']:6d}  {r['n_triangles']:5d}  "
              f"{r['beta_1']:3d}  {r['energy_gradient_pct']:6.1f}  "
              f"{r['energy_curl_pct']:6.1f}  {r['energy_harmonic_pct']:6.1f}  "
              f"{r['correlation_s_lr']['rho']:+8.3f}  "
              f"{r['correlation_s_acc']['rho']:+9.3f}")

    # Save
    out_path = os.path.join(ROOT, "results", "hodge_attribution.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
