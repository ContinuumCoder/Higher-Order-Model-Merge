#!/usr/bin/env python3
"""
Topology-Aware Alignment: align models along minimum-harmonic MST
instead of star topology to a single reference.

Hypothesis: standard star alignment accumulates harmonic residuals.
Aligning along paths of minimum harmonic energy should reduce these residuals.

Either outcome is a finding:
  - Harmonic decreases → alignment topology matters, Hodge guides better alignment
  - Harmonic unchanged → harmonic is intrinsic, not an alignment artifact
"""

import glob
import json
import os
import sys
import time
from collections import OrderedDict, deque
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.zoo.plain_cnn import PlainCNN, get_perm_spec_plaincnn
from src.barriers.align import compute_weight_matching_perm, apply_permutation
from src.topology.hodge import (
    build_mergeability_complex, build_boundary_operators,
    hodge_analysis, hodge_decomposition,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_eval_loader(data_dir="data", subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    return DataLoader(Subset(ds, list(range(min(subset_size, len(ds))))),
                      batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    model.train()
    for x, _ in loader:
        model(x.to(device))
    model.eval()


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += F.cross_entropy(model(x), y, reduction='sum').item()
        n += x.size(0)
    return total / n


def align_pair(sd_ref, sd_target):
    """Align target to ref using PlainCNN permutation spec."""
    groups = get_perm_spec_plaincnn()
    sd_aligned = OrderedDict({k: v.clone() for k, v in sd_target.items()})
    for g in groups:
        perm = compute_weight_matching_perm(sd_ref, sd_aligned, g)
        sd_aligned = apply_permutation(sd_aligned, g, perm)
    return sd_aligned


def compute_pairwise_barriers(sds, names, device, loader, n_interp=5):
    """Fast pairwise barrier computation."""
    n = len(sds)
    model = PlainCNN().to(device)
    alphas = np.linspace(0, 1, n_interp)

    # Endpoint losses
    endpoint_losses = np.zeros(n)
    for i in range(n):
        model.load_state_dict(sds[i])
        recalibrate_bn(model, loader, device)
        endpoint_losses[i] = eval_loss(model, loader, device)

    barrier_matrix = np.zeros((n, n))
    pairs = list(combinations(range(n), 2))
    for idx, (i, j) in enumerate(pairs):
        max_loss = max(endpoint_losses[i], endpoint_losses[j])
        for alpha in alphas[1:-1]:
            sd_interp = {k: (1 - alpha) * sds[i][k] + alpha * sds[j][k] for k in sds[i]}
            model.load_state_dict(sd_interp)
            recalibrate_bn(model, loader, device)
            loss = eval_loss(model, loader, device)
            max_loss = max(max_loss, loss)
        barrier_matrix[i, j] = barrier_matrix[j, i] = max_loss - max(endpoint_losses[i], endpoint_losses[j])
        if (idx + 1) % 30 == 0 or idx + 1 == len(pairs):
            print(f"    {idx+1}/{len(pairs)} pairs")

    return barrier_matrix, endpoint_losses


def compute_hodge_at_tau(barrier_matrix, triplet_barriers, tau, n):
    """Compute Hodge decomposition at given tau, return per-edge decomposition."""
    V, E, T = build_mergeability_complex(barrier_matrix, triplet_barriers, tau)
    if len(E) < 2:
        return None

    B1, B2 = build_boundary_operators(V, E, T)
    f = np.array([barrier_matrix[e[0], e[1]] for e in E])
    decomp = hodge_decomposition(B1, B2, f)
    ha = hodge_analysis(B1, B2)

    total_energy = max(decomp['norm_f']**2, 1e-12)
    return {
        'edges': E, 'triangles': T,
        'beta_1': ha['beta_1'],
        'gradient': decomp['gradient'],
        'curl': decomp['curl'],
        'harmonic': decomp['harmonic'],
        'grad_pct': 100 * decomp['norm_gradient']**2 / total_energy,
        'curl_pct': 100 * decomp['norm_curl']**2 / total_energy,
        'harm_pct': 100 * decomp['norm_harmonic']**2 / total_energy,
    }


def build_harmonic_graph(n, edges, harmonic_vec):
    """Build per-edge harmonic energy matrix."""
    H = np.zeros((n, n))
    for idx, (i, j) in enumerate(edges):
        h_energy = harmonic_vec[idx]**2
        H[i, j] = H[j, i] = h_energy
    return H


def mst_bfs_order(mst_matrix, root):
    """BFS on MST from root, return list of (parent, child) pairs."""
    n = mst_matrix.shape[0]
    # Make symmetric
    full = mst_matrix + mst_matrix.T
    visited = set([root])
    queue = deque([root])
    order = []

    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if neighbor not in visited and full[node, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)
                order.append((node, neighbor))  # align child to parent

    return order


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    device = torch.device(args.device)
    loader = get_eval_loader(args.data_dir)

    # =========================================================================
    # Load original (unaligned) checkpoints
    # =========================================================================
    print("Loading original checkpoints...")
    paths = sorted(glob.glob(os.path.join(ROOT, "checkpoints_plain", "plaincnn_s*_lr*_wd0.0001_augbasic.pt")))
    # Use only the 24 models (8 seeds × 3 LRs)
    sds_raw = []
    names = []
    for p_path in paths:
        ckpt = torch.load(p_path, map_location='cpu', weights_only=False)
        sds_raw.append({k: v.float() for k, v in ckpt['model_state_dict'].items()})
        names.append(os.path.basename(p_path))

    n = len(sds_raw)
    short_names = [sn.replace("plaincnn_", "").replace("_wd0.0001_augbasic.pt", "") for sn in names]
    seeds = [int(s.split("_")[0].replace("s", "")) for s in short_names]
    print(f"  {n} models: {short_names}")

    # =========================================================================
    # STEP 1: Load existing star alignment results (already computed)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Load Existing Star Alignment (from results_24/)")
    print(f"{'='*60}")

    star_data = np.load(os.path.join(ROOT, "results_24", "barriers_plain.npz"), allow_pickle=True)
    barrier_star = star_data["barrier_matrix"]
    ep_star = star_data["endpoint_losses"]
    triplet_barriers_star = {}
    for key, val in zip(star_data["triplet_keys"], star_data["triplet_vals"]):
        triplet_barriers_star[tuple(sorted(key))] = float(val)

    upper_star = barrier_star[np.triu_indices(n, k=1)]
    print(f"  Loaded {len(triplet_barriers_star)} triplet barriers")
    print(f"  Barrier stats: min={upper_star.min():.4f} median={np.median(upper_star):.4f} "
          f"mean={upper_star.mean():.4f} max={upper_star.max():.4f}")

    model = PlainCNN().to(device)
    from run_full_pipeline import barycentric_grid
    grid = barycentric_grid(resolution=4)
    interior = grid[(grid > 1e-8).all(axis=1)]

    # Hodge decomposition for star alignment
    tau_hodge = np.median(upper_star)
    hodge_star = compute_hodge_at_tau(barrier_star, triplet_barriers_star, tau_hodge, n)
    print(f"\n  Star Hodge at τ={tau_hodge:.4f}:")
    print(f"    β₁={hodge_star['beta_1']}, gradient={hodge_star['grad_pct']:.1f}%, "
          f"curl={hodge_star['curl_pct']:.1f}%, harmonic={hodge_star['harm_pct']:.1f}%")

    # =========================================================================
    # STEP 2: Build harmonic MST
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Build Harmonic MST")
    print(f"{'='*60}")

    H = build_harmonic_graph(n, hodge_star['edges'], hodge_star['harmonic'])

    # For edges not in the complex, use total barrier as fallback (high weight)
    # This ensures MST covers all nodes
    for i, j in combinations(range(n), 2):
        if H[i, j] == 0:
            H[i, j] = H[j, i] = barrier_star[i, j]**2  # large weight for non-complex edges

    mst = minimum_spanning_tree(csr_matrix(H))
    mst_dense = mst.toarray()

    # Find root: model with lowest total harmonic energy
    harm_per_model = np.zeros(n)
    for idx, (i, j) in enumerate(hodge_star['edges']):
        h2 = hodge_star['harmonic'][idx]**2
        harm_per_model[i] += h2
        harm_per_model[j] += h2
    root = int(np.argmin(harm_per_model))
    print(f"  Root (lowest harmonic): {short_names[root]} (seed={seeds[root]})")

    alignment_order = mst_bfs_order(mst_dense, root)
    print(f"  MST alignment order ({len(alignment_order)} edges):")
    for parent, child in alignment_order:
        h_weight = H[parent, child]
        print(f"    {short_names[parent]:>15s} → {short_names[child]:<15s}  (h²={h_weight:.4f})")

    # =========================================================================
    # STEP 3: MST Alignment
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: MST Alignment")
    print(f"{'='*60}")
    t0 = time.time()

    # Start from raw checkpoints, align along MST
    groups = get_perm_spec_plaincnn()
    sds_mst = [None] * n
    sds_mst[root] = OrderedDict({k: v.clone() for k, v in sds_raw[root].items()})

    for parent, child in alignment_order:
        # Align child's raw weights to parent's already-aligned weights
        sd_child = OrderedDict({k: v.clone() for k, v in sds_raw[child].items()})
        for g in groups:
            perm = compute_weight_matching_perm(sds_mst[parent], sd_child, g)
            sd_child = apply_permutation(sd_child, g, perm)
        sds_mst[child] = sd_child

    print(f"  Aligned in {time.time()-t0:.1f}s")

    # =========================================================================
    # STEP 4: Recompute barriers for MST alignment
    # =========================================================================
    print("  Computing pairwise barriers (MST)...")
    barrier_mst, ep_mst = compute_pairwise_barriers(sds_mst, short_names, device, loader)
    upper_mst = barrier_mst[np.triu_indices(n, k=1)]
    print(f"  Barrier stats: min={upper_mst.min():.4f} median={np.median(upper_mst):.4f} "
          f"mean={upper_mst.mean():.4f} max={upper_mst.max():.4f}")

    # Triplet barriers for MST alignment (same cliques)
    tau75_mst = np.percentile(upper_mst, 75)
    edge_set_mst = set()
    for i, j in combinations(range(n), 2):
        if barrier_mst[i, j] <= tau75_mst:
            edge_set_mst.add(frozenset({i, j}))
    cliques_mst = []
    for triple in combinations(range(n), 3):
        i, j, k = triple
        if frozenset({i,j}) in edge_set_mst and frozenset({i,k}) in edge_set_mst and frozenset({j,k}) in edge_set_mst:
            cliques_mst.append(triple)

    print(f"  Computing triplet barriers ({len(cliques_mst)} cliques at τ={tau75_mst:.4f})...")
    triplet_barriers_mst = {}
    for idx, (i, j, k) in enumerate(cliques_mst):
        base_loss = max(ep_mst[i], ep_mst[j], ep_mst[k])
        worst_loss = base_loss
        for lam in interior:
            sd_interp = {key: lam[0]*sds_mst[i][key] + lam[1]*sds_mst[j][key] + lam[2]*sds_mst[k][key]
                         for key in sds_mst[i]}
            model.load_state_dict(sd_interp)
            recalibrate_bn(model, loader, device)
            loss = eval_loss(model, loader, device)
            worst_loss = max(worst_loss, loss)
        triplet_barriers_mst[(i,j,k)] = worst_loss - base_loss
        if (idx+1) % 50 == 0 or idx+1 == len(cliques_mst):
            print(f"    {idx+1}/{len(cliques_mst)}")

    # Hodge for MST
    tau_hodge_mst = np.median(upper_mst)
    hodge_mst = compute_hodge_at_tau(barrier_mst, triplet_barriers_mst, tau_hodge_mst, n)
    print(f"\n  MST Hodge at τ={tau_hodge_mst:.4f}:")
    print(f"    β₁={hodge_mst['beta_1']}, gradient={hodge_mst['grad_pct']:.1f}%, "
          f"curl={hodge_mst['curl_pct']:.1f}%, harmonic={hodge_mst['harm_pct']:.1f}%")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: Star vs MST Alignment")
    print(f"{'='*70}")
    print(f"  {'':>25s}  {'Star':>10s}  {'MST':>10s}  {'Δ':>10s}")
    print(f"  {'Barrier mean':>25s}  {upper_star.mean():10.4f}  {upper_mst.mean():10.4f}  "
          f"{upper_mst.mean()-upper_star.mean():+10.4f}")
    print(f"  {'Barrier median':>25s}  {np.median(upper_star):10.4f}  {np.median(upper_mst):10.4f}  "
          f"{np.median(upper_mst)-np.median(upper_star):+10.4f}")
    print(f"  {'Barrier max':>25s}  {upper_star.max():10.4f}  {upper_mst.max():10.4f}  "
          f"{upper_mst.max()-upper_star.max():+10.4f}")
    if hodge_star and hodge_mst:
        print(f"  {'β₁':>25s}  {hodge_star['beta_1']:10d}  {hodge_mst['beta_1']:10d}  "
              f"{hodge_mst['beta_1']-hodge_star['beta_1']:+10d}")
        print(f"  {'Gradient %':>25s}  {hodge_star['grad_pct']:10.1f}  {hodge_mst['grad_pct']:10.1f}  "
              f"{hodge_mst['grad_pct']-hodge_star['grad_pct']:+10.1f}")
        print(f"  {'Curl %':>25s}  {hodge_star['curl_pct']:10.1f}  {hodge_mst['curl_pct']:10.1f}  "
              f"{hodge_mst['curl_pct']-hodge_star['curl_pct']:+10.1f}")
        print(f"  {'Harmonic %':>25s}  {hodge_star['harm_pct']:10.1f}  {hodge_mst['harm_pct']:10.1f}  "
              f"{hodge_mst['harm_pct']-hodge_star['harm_pct']:+10.1f}")

    # Per-pair comparison
    improved = np.sum(barrier_mst[np.triu_indices(n, k=1)] < barrier_star[np.triu_indices(n, k=1)])
    worsened = np.sum(barrier_mst[np.triu_indices(n, k=1)] > barrier_star[np.triu_indices(n, k=1)])
    total_pairs = n * (n - 1) // 2
    print(f"\n  Per-pair: {improved} improved, {worsened} worsened, "
          f"{total_pairs-improved-worsened} unchanged (of {total_pairs})")

    # Save
    out = {
        "star": {
            "barrier_mean": float(upper_star.mean()),
            "barrier_median": float(np.median(upper_star)),
            "barrier_max": float(upper_star.max()),
            "beta_1": hodge_star['beta_1'] if hodge_star else None,
            "grad_pct": hodge_star['grad_pct'] if hodge_star else None,
            "curl_pct": hodge_star['curl_pct'] if hodge_star else None,
            "harm_pct": hodge_star['harm_pct'] if hodge_star else None,
        },
        "mst": {
            "barrier_mean": float(upper_mst.mean()),
            "barrier_median": float(np.median(upper_mst)),
            "barrier_max": float(upper_mst.max()),
            "beta_1": hodge_mst['beta_1'] if hodge_mst else None,
            "grad_pct": hodge_mst['grad_pct'] if hodge_mst else None,
            "curl_pct": hodge_mst['curl_pct'] if hodge_mst else None,
            "harm_pct": hodge_mst['harm_pct'] if hodge_mst else None,
        },
        "root": short_names[root],
        "mst_order": [(short_names[p], short_names[c]) for p, c in alignment_order],
        "pairs_improved": int(improved),
        "pairs_worsened": int(worsened),
    }
    out_path = os.path.join(ROOT, "results", "topo_alignment.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save barrier matrices for further analysis
    np.savez(os.path.join(ROOT, "results", "barriers_star_vs_mst.npz"),
             barrier_star=barrier_star, barrier_mst=barrier_mst,
             names=np.array(names))
    print("Barrier matrices saved to results/barriers_star_vs_mst.npz")


if __name__ == "__main__":
    main()
