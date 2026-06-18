#!/usr/bin/env python3
"""
Full pipeline for PlainCNN: align -> barriers -> Hodge analysis -> figures.
"""

import glob
import json
import os
import sys
import time
from collections import OrderedDict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))

from src.zoo.plain_cnn import PlainCNN, get_perm_spec_plaincnn
from src.barriers.align import compute_weight_matching_perm, apply_permutation
from src.topology.hodge import (
    build_mergeability_complex, build_boundary_operators,
    hodge_analysis, hodge_decomposition, filtration_analysis
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_eval_loader(data_dir, subset_size=5000):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    subset = Subset(ds, list(range(min(subset_size, len(ds)))))
    return DataLoader(subset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


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


def load_state_dicts(ckpt_dir, pre_aligned=False):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "plaincnn_*.pt")))
    sds, names, accs = [], [], []
    for p in paths:
        ckpt = torch.load(p, map_location='cpu', weights_only=not pre_aligned)
        if pre_aligned:
            # Aligned checkpoints are raw state_dicts (OrderedDict)
            sds.append({k: v.float() for k, v in ckpt.items()})
            accs.append(0)
        else:
            sds.append({k: v.float() for k, v in ckpt['model_state_dict'].items()})
            accs.append(ckpt.get('test_acc', 0))
        names.append(os.path.basename(p))
    return sds, names, accs


def align_all(sds, ref_idx=0):
    """Align all state dicts to reference using weight matching."""
    groups = get_perm_spec_plaincnn()
    ref = sds[ref_idx]
    aligned = []
    for i, sd in enumerate(sds):
        if i == ref_idx:
            aligned.append(OrderedDict({k: v.clone() for k, v in sd.items()}))
        else:
            sd_a = OrderedDict({k: v.clone() for k, v in sd.items()})
            for g in groups:
                perm = compute_weight_matching_perm(ref, sd_a, g)
                sd_a = apply_permutation(sd_a, g, perm)
            aligned.append(sd_a)
    return aligned


def interpolate(sd_a, sd_b, alpha):
    return {k: (1-alpha)*sd_a[k] + alpha*sd_b[k] for k in sd_a}


def interpolate_bary(sds, lambdas):
    result = {}
    for k in sds[0]:
        result[k] = sum(l * sd[k].float() for l, sd in zip(lambdas, sds))
    return result


def barycentric_grid(resolution=5):
    pts = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            pts.append([i/resolution, j/resolution, k/resolution])
    return np.array(pts)


def load_checkpoint(ckpt_path):
    """Load incremental checkpoint if it exists."""
    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        ckpt = {
            "endpoint_losses": data["endpoint_losses"],
            "barrier_matrix": data["barrier_matrix"],
            "pairwise_done": set(map(tuple, data["pairwise_done"])) if len(data["pairwise_done"]) > 0 else set(),
            "triplet_keys": [tuple(k) for k in data["triplet_keys"]] if len(data["triplet_keys"]) > 0 else [],
            "triplet_vals": list(data["triplet_vals"]) if len(data["triplet_vals"]) > 0 else [],
            "stage": str(data["stage"]) if "stage" in data else "computing",
        }
        print(f"  Resumed from checkpoint: {len(ckpt['pairwise_done'])} pairs, "
              f"{len(ckpt['triplet_keys'])} triplets done, stage={ckpt['stage']}")
        return ckpt
    return None


def save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names, stage="computing"):
    """Save incremental checkpoint (atomic write via temp file)."""
    if triplet_barriers:
        tk = np.array(list(triplet_barriers.keys()), dtype=np.int64)
        tv = np.array(list(triplet_barriers.values()), dtype=np.float64)
    else:
        tk = np.array([], dtype=np.int64).reshape(0, 3)
        tv = np.array([], dtype=np.float64)
    pw_done = np.array(sorted(pairwise_done), dtype=np.int64) if pairwise_done else np.array([], dtype=np.int64).reshape(0, 2)
    tmp_path = ckpt_path + ".tmp"
    np.savez(tmp_path, endpoint_losses=endpoint_losses, barrier_matrix=barrier_matrix,
             pairwise_done=pw_done, triplet_keys=tk, triplet_vals=tv,
             model_names=np.array(names), stage=np.array(stage))
    # np.savez auto-appends .npz if missing
    actual_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"
    os.replace(actual_tmp, ckpt_path)


def _triplet_worker(gpu_id, work_queue, result_queue, ckpt_dir, endpoint_losses_list,
                    data_dir, bn_samples, eval_samples):
    """Worker process: compute triplet barriers on one GPU."""
    device = torch.device(f'cuda:{gpu_id}')

    # Load aligned state dicts from disk (each worker loads independently)
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "plaincnn_*.pt")))
    sds = []
    for p in paths:
        sd = torch.load(p, map_location='cpu', weights_only=True)
        sds.append({k: v.float() for k, v in sd.items()})

    model = PlainCNN().to(device)
    endpoint_losses = np.array(endpoint_losses_list)

    # Loaders
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    bn_loader = DataLoader(Subset(train_ds, list(range(bn_samples))),
                           batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(Subset(train_ds, list(range(eval_samples))),
                             batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    grid = barycentric_grid(resolution=5)
    interior = grid[(grid > 1e-8).all(axis=1)]

    while True:
        item = work_queue.get()
        if item is None:  # poison pill
            break
        i, j, k = item
        base_loss = max(endpoint_losses[i], endpoint_losses[j], endpoint_losses[k])
        worst_loss = base_loss
        for lam in interior:
            sd_interp = {key: lam[0]*sds[i][key] + lam[1]*sds[j][key] + lam[2]*sds[k][key]
                         for key in sds[i]}
            model.load_state_dict(sd_interp)
            model.train()
            with torch.no_grad():
                for x, _ in bn_loader:
                    model(x.to(device))
            model.eval()
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(device), y.to(device)
                    total_loss += F.cross_entropy(model(x), y, reduction='sum').item()
                    n += x.size(0)
            worst_loss = max(worst_loss, total_loss / n)
        result_queue.put(((i, j, k), worst_loss - base_loss))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints_plain")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--n-interp", type=int, default=11)
    p.add_argument("--use-aligned", action="store_true",
                   help="Load pre-aligned checkpoints (skip alignment step)")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Number of GPUs for parallel triplet computation")
    p.add_argument("--bn-samples", type=int, default=1000,
                   help="Samples for BN recalibration (default 1000, was 5000)")
    p.add_argument("--eval-samples", type=int, default=2000,
                   help="Samples for loss evaluation (default 2000, was 5000)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    device = torch.device(args.device)
    ckpt_path = os.path.join(args.output_dir, "checkpoint.npz")

    # Load
    print("Loading checkpoints...")
    sds, names, accs = load_state_dicts(args.ckpt_dir, pre_aligned=args.use_aligned)
    n = len(sds)
    print(f"  {n} models: {names}")
    if not args.use_aligned:
        print(f"  Accuracies: {[f'{a:.1f}%' for a in accs]}")

    # Align
    if args.use_aligned:
        print("\nUsing pre-aligned checkpoints, skipping alignment.")
        sds_aligned = sds
    else:
        print("\nAligning all models to reference (model 0)...")
        t0 = time.time()
        sds_aligned = align_all(sds, ref_idx=0)
        print(f"  Done in {time.time()-t0:.1f}s")

    # Try to resume from checkpoint
    ckpt = load_checkpoint(ckpt_path)

    # Eval loader
    loader = get_eval_loader(args.data_dir, subset_size=5000)
    model = PlainCNN().to(device)

    # Endpoint losses
    if ckpt is not None and len(ckpt["endpoint_losses"]) == n:
        endpoint_losses = ckpt["endpoint_losses"]
        print(f"\nEndpoint losses (from checkpoint):")
        for i in range(n):
            print(f"  {names[i]}: loss={endpoint_losses[i]:.4f}")
    else:
        print("\nEndpoint losses:")
        endpoint_losses = []
        for i in range(n):
            model.load_state_dict(sds_aligned[i])
            model.to(device)
            recalibrate_bn(model, loader, device)
            loss = eval_loss(model, loader, device)
            endpoint_losses.append(loss)
            print(f"  {names[i]}: loss={loss:.4f}")
        endpoint_losses = np.array(endpoint_losses)

    # Pairwise barriers (with resume)
    alphas = np.linspace(0, 1, args.n_interp)
    if ckpt is not None:
        barrier_matrix = ckpt["barrier_matrix"]
        pairwise_done = ckpt["pairwise_done"]
    else:
        barrier_matrix = np.zeros((n, n))
        pairwise_done = set()

    num_pairs = n * (n - 1) // 2
    num_remaining = num_pairs - len(pairwise_done)
    print(f"\nComputing pairwise barriers: {len(pairwise_done)}/{num_pairs} done, {num_remaining} remaining...")
    t0 = time.time()
    computed = 0
    for i, j in combinations(range(n), 2):
        if (i, j) in pairwise_done:
            continue
        max_loss = max(endpoint_losses[i], endpoint_losses[j])
        for alpha in alphas[1:-1]:
            sd_interp = interpolate(sds_aligned[i], sds_aligned[j], alpha)
            model.load_state_dict(sd_interp)
            model.to(device)
            recalibrate_bn(model, loader, device)
            loss = eval_loss(model, loader, device)
            max_loss = max(max_loss, loss)
        barrier = max_loss - max(endpoint_losses[i], endpoint_losses[j])
        barrier_matrix[i, j] = barrier_matrix[j, i] = barrier
        pairwise_done.add((i, j))
        computed += 1
        if computed % 20 == 0 or computed == num_remaining:
            elapsed = time.time() - t0
            total_done = len(pairwise_done)
            print(f"  {total_done}/{num_pairs} ({elapsed:.0f}s)")
            save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, {}, names)
    if computed > 0:
        save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, {}, names)
    print(f"Pairwise barriers done ({time.time()-t0:.0f}s for {computed} new pairs)")

    # Barrier stats
    upper = barrier_matrix[np.triu_indices(n, k=1)]
    print(f"\nBarrier stats: min={upper.min():.4f} median={np.median(upper):.4f} "
          f"mean={upper.mean():.4f} max={upper.max():.4f}")

    # Triplet barriers: compute for 3-cliques at various tau
    percentiles = [25, 50, 75, 90]
    for pct in percentiles:
        tau = np.percentile(upper, pct)
        n_edges = np.sum(upper <= tau)
        print(f"  tau={tau:.4f} (p{pct}): {n_edges}/{len(upper)} edges")

    tau_triplet = np.percentile(upper, 75)
    edge_set = set()
    for i, j in combinations(range(n), 2):
        if barrier_matrix[i, j] <= tau_triplet:
            edge_set.add(frozenset({i, j}))

    # Find 3-cliques
    cliques = []
    for triple in combinations(range(n), 3):
        i, j, k = triple
        if (frozenset({i,j}) in edge_set and frozenset({i,k}) in edge_set
                and frozenset({j,k}) in edge_set):
            cliques.append(triple)
    print(f"\n3-cliques at tau={tau_triplet:.4f}: {len(cliques)}")

    # Resume triplet barriers from checkpoint
    triplet_barriers = {}
    if ckpt is not None:
        for k, v in zip(ckpt["triplet_keys"], ckpt["triplet_vals"]):
            triplet_barriers[tuple(k)] = float(v)

    # Compute triplet barriers (with resume + multi-GPU)
    grid = barycentric_grid(resolution=5)
    interior = grid[(grid > 1e-8).all(axis=1)]
    remaining_cliques = [c for c in cliques if c not in triplet_barriers]
    print(f"Computing triplet barriers: {len(triplet_barriers)} done, {len(remaining_cliques)} remaining "
          f"({len(interior)} interior points each, {args.n_gpus} GPU(s), "
          f"bn={args.bn_samples}, eval={args.eval_samples})...")
    t0 = time.time()

    if remaining_cliques and args.n_gpus > 1:
        # ---- Multi-GPU parallel triplet computation ----
        work_queue = mp.Queue()
        result_queue = mp.Queue()
        for c in remaining_cliques:
            work_queue.put(c)
        for _ in range(args.n_gpus):
            work_queue.put(None)  # poison pills

        workers = []
        for gpu_id in range(args.n_gpus):
            w = mp.Process(target=_triplet_worker,
                           args=(gpu_id, work_queue, result_queue, args.ckpt_dir,
                                 endpoint_losses.tolist(), args.data_dir,
                                 args.bn_samples, args.eval_samples))
            w.start()
            workers.append(w)

        completed = 0
        while completed < len(remaining_cliques):
            triplet, barrier = result_queue.get()
            triplet_barriers[triplet] = barrier
            completed += 1
            if completed % 20 == 0 or completed == len(remaining_cliques):
                elapsed = time.time() - t0
                print(f"  {len(triplet_barriers)}/{len(cliques)} ({elapsed:.0f}s)")
                save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names)

        for w in workers:
            w.join()

    elif remaining_cliques:
        # ---- Single-GPU sequential (with reduced samples) ----
        bn_loader = get_eval_loader(args.data_dir, subset_size=args.bn_samples)
        eval_loader_small = get_eval_loader(args.data_dir, subset_size=args.eval_samples)
        for idx, (i, j, k) in enumerate(remaining_cliques):
            base_loss = max(endpoint_losses[i], endpoint_losses[j], endpoint_losses[k])
            worst_loss = base_loss
            for lam in interior:
                sd_interp = interpolate_bary([sds_aligned[i], sds_aligned[j], sds_aligned[k]], lam)
                model.load_state_dict(sd_interp)
                model.to(device)
                recalibrate_bn(model, bn_loader, device)
                loss = eval_loss(model, eval_loader_small, device)
                worst_loss = max(worst_loss, loss)
            triplet_barriers[(i,j,k)] = worst_loss - base_loss
            if (idx+1) % 10 == 0 or idx+1 == len(remaining_cliques):
                elapsed = time.time() - t0
                print(f"  {len(triplet_barriers)}/{len(cliques)} ({elapsed:.0f}s)")
                save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names)

    if remaining_cliques:
        save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names)
    print(f"Triplet barriers done ({time.time()-t0:.0f}s for {len(remaining_cliques)} new triplets)")

    # Discordant analysis
    if triplet_barriers:
        print("\n=== Discordant Triplets (top 20) ===")
        sorted_tb = sorted(triplet_barriers.items(), key=lambda x: -x[1])
        for (i,j,k), tb in sorted_tb[:20]:
            pw = max(barrier_matrix[i,j], barrier_matrix[i,k], barrier_matrix[j,k])
            pw_min = min(barrier_matrix[i,j], barrier_matrix[i,k], barrier_matrix[j,k])
            ratio = tb / max(pw, 1e-8)
            flag = " *** DISCORDANT ***" if tb > pw * 1.2 else ""
            print(f"  ({i},{j},{k}): pw_max={pw:.4f} pw_min={pw_min:.4f} triplet={tb:.4f} ratio={ratio:.2f}x{flag}")

    # Save final barrier data (separate from checkpoint)
    if triplet_barriers:
        tk = np.array(list(triplet_barriers.keys()), dtype=np.int64)
        tv = np.array(list(triplet_barriers.values()), dtype=np.float64)
    else:
        tk = np.array([], dtype=np.int64).reshape(0, 3)
        tv = np.array([], dtype=np.float64)

    barrier_path = os.path.join(args.output_dir, "barriers_plain.npz")
    np.savez(barrier_path, barrier_matrix=barrier_matrix, endpoint_losses=endpoint_losses,
             triplet_keys=tk, triplet_vals=tv, model_names=np.array(names))
    print(f"\nSaved barriers to {barrier_path}")
    save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names, stage="barriers_done")

    # =========================================================================
    # Hodge analysis (skip if already done)
    # =========================================================================
    completed_stage = ckpt["stage"] if ckpt is not None else "computing"
    report_path = os.path.join(args.output_dir, "report.json")

    if completed_stage == "all_done" and os.path.exists(report_path):
        print("\nAll stages already completed (checkpoint says all_done). Nothing to do.")
        print("DONE!")
        return

    print("\n" + "=" * 60)
    print("HODGE ANALYSIS")
    print("=" * 60)

    tau_range = np.linspace(0, upper.max() * 1.1, 80)
    results = filtration_analysis(barrier_matrix, triplet_barriers, tau_range)

    print(f"\n{'tau':>8s} {'β₀':>4s} {'β₁':>4s} {'edges':>6s} {'tri':>5s} {'fill':>6s}")
    print("-" * 40)
    prev_b1 = -1
    for r in results:
        if r['beta_1'] != prev_b1 or r == results[0] or r == results[-1] or r['n_edges'] in [1, n*(n-1)//2]:
            print(f"{r['tau']:8.4f} {r['beta_0']:4d} {r['beta_1']:4d} "
                  f"{r['n_edges']:6d} {r['n_triangles']:5d} {r['fill_ratio']:6.3f}")
            prev_b1 = r['beta_1']

    beta1_max = max(r['beta_1'] for r in results)
    beta1_nonzero = [r for r in results if r['beta_1'] > 0]
    print(f"\nmax β₁ = {beta1_max}")
    if beta1_nonzero:
        print(f"β₁ > 0 in tau range [{beta1_nonzero[0]['tau']:.4f}, {beta1_nonzero[-1]['tau']:.4f}]")
    else:
        print("β₁ = 0 everywhere")

    # Hodge decomposition at interesting tau
    interesting = [r for r in results if r['n_edges'] > 3 and r['n_triangles'] < r['n_edges']]
    if interesting:
        mid = interesting[len(interesting)//2]
        tau_h = mid['tau']
        V, E, T = build_mergeability_complex(barrier_matrix, triplet_barriers, tau_h)
        B1, B2 = build_boundary_operators(V, E, T)
        f = np.array([barrier_matrix[e[0], e[1]] for e in E])
        decomp = hodge_decomposition(B1, B2, f)
        total_var = max(decomp['norm_f']**2, 1e-12)
        print(f"\nHodge decomposition at τ={tau_h:.4f} ({len(E)} edges, {len(T)} tri):")
        print(f"  gradient: {100*decomp['norm_gradient']**2/total_var:.1f}%")
        print(f"  curl:     {100*decomp['norm_curl']**2/total_var:.1f}%")
        print(f"  harmonic: {100*decomp['norm_harmonic']**2/total_var:.1f}%")

    save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names, stage="hodge_done")

    # =========================================================================
    # Figures
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(args.output_dir, "figures")

    # Fig 1: Barrier matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(barrier_matrix, cmap='YlOrRd', aspect='equal')
    plt.colorbar(im, ax=ax, label='Merge barrier')
    short_names = [n.replace('plaincnn_', '').replace('.pt', '') for n in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title('Pairwise Merge Barriers (aligned)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'barrier_matrix.png'), dpi=150)
    plt.close()
    print("  Saved barrier_matrix.png")

    # Fig 2: Filtration curves
    taus = [r['tau'] for r in results]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(taus, [r['beta_0'] for r in results], 'b-o', ms=2, label='β₀ (components)')
    ax1.plot(taus, [r['beta_1'] for r in results], 'r-s', ms=2, label='β₁ (1-holes)')
    ax1.set_ylabel('Betti number')
    ax1.legend()
    ax1.set_title('Filtration of Mergeability Complex')

    ax2.plot(taus, [r['fill_ratio'] for r in results], 'g-^', ms=2)
    ax2.set_ylabel('Triangle fill ratio')
    ax2.set_xlabel('Threshold τ')
    ax2.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'filtration.png'), dpi=150)
    plt.close()
    print("  Saved filtration.png")

    # Fig 3: Hodge spectrum
    if interesting:
        eigs = np.sort(np.linalg.eigvalsh(
            (B1.T @ B1 + B2 @ B2.T).toarray()))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.stem(range(len(eigs)), eigs, linefmt='b-', markerfmt='bo', basefmt='k-')
        ax.axhline(y=1e-8, color='r', linestyle='--', alpha=0.5, label=f'β₁={int(np.sum(eigs < 1e-8))} zero eigenvalues')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue of Δ₁')
        ax.set_title(f'Hodge Spectrum at τ={tau_h:.4f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'hodge_spectrum.png'), dpi=150)
        plt.close()
        print("  Saved hodge_spectrum.png")

    save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names, stage="figures_done")

    # Save JSON report
    report = {
        "n_models": n, "model_names": names, "accuracies": accs,
        "barrier_stats": {"min": float(upper.min()), "median": float(np.median(upper)),
                          "mean": float(upper.mean()), "max": float(upper.max())},
        "n_3cliques": len(cliques), "n_triplet_barriers": len(triplet_barriers),
        "beta1_max": beta1_max,
        "filtration": [{k: v for k, v in r.items() if k != 'eigenvalues_L1'}
                       for r in results],
    }
    with open(os.path.join(args.output_dir, "report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nReport saved to {args.output_dir}/report.json")

    save_checkpoint(ckpt_path, endpoint_losses, barrier_matrix, pairwise_done, triplet_barriers, names, stage="all_done")
    print("DONE!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
