#!/usr/bin/env python3
"""
Hodge analysis on the mergeability complex.

Builds the simplicial complex from barrier data, computes boundary operators,
Hodge Laplacians, Betti numbers, and the Hodge decomposition of 1-cochains.

Usage:
    python hodge.py --barriers barriers.npz --output hodge_results.json
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Build the mergeability complex
# ---------------------------------------------------------------------------

def build_mergeability_complex(barrier_matrix, triplet_barriers, tau):
    """Construct the simplicial complex at threshold tau.

    Args:
        barrier_matrix: (n, n) symmetric matrix of pairwise barriers.
        triplet_barriers: dict mapping (i,j,k) -> triplet barrier value.
        tau: inclusion threshold.

    Returns:
        vertices: sorted list of vertex indices.
        edges: list of sorted tuples (i, j) with barrier <= tau.
        triangles: list of sorted tuples (i, j, k) with all pairwise barriers
                   <= tau AND triplet barrier <= tau.
    """
    n = barrier_matrix.shape[0]
    vertices = list(range(n))

    edges = []
    for i, j in combinations(range(n), 2):
        if barrier_matrix[i, j] <= tau:
            edges.append((i, j))

    # A triangle requires all 3 edges present AND triplet barrier <= tau
    edge_set = set(edges)
    triangles = []
    for triple in combinations(range(n), 3):
        i, j, k = triple
        if ((i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set):
            key = tuple(sorted(triple))
            if key in triplet_barriers and triplet_barriers[key] <= tau:
                triangles.append(key)

    return vertices, edges, triangles


# ---------------------------------------------------------------------------
# Boundary operators
# ---------------------------------------------------------------------------

def build_boundary_operators(vertices, edges, triangles):
    """Build sparse boundary operators B1 and B2.

    B1: edges -> vertices, shape (|V|, |E|)
        B1[:, e] has +1 at head, -1 at tail for edge e = (i, j), i < j.
        Convention: B1[j, e] = +1, B1[i, e] = -1.

    B2: triangles -> edges, shape (|E|, |T|)
        B2[:, t] encodes the boundary of triangle t = (i, j, k).
        Signs follow the standard orientation: +(j,k) - (i,k) + (i,j).

    Returns:
        B1: sparse matrix of shape (n_vertices, n_edges)
        B2: sparse matrix of shape (n_edges, n_triangles)
    """
    n_v = len(vertices)
    n_e = len(edges)
    n_t = len(triangles)

    # Vertex index map (usually identity but let's be safe)
    v_idx = {v: idx for idx, v in enumerate(vertices)}

    # Edge index map
    e_idx = {e: idx for idx, e in enumerate(edges)}

    # B1: (n_v x n_e)
    if n_e > 0:
        rows, cols, vals = [], [], []
        for idx, (i, j) in enumerate(edges):
            # Convention: column for edge (i,j) has -1 at row i, +1 at row j
            rows.extend([v_idx[i], v_idx[j]])
            cols.extend([idx, idx])
            vals.extend([-1.0, 1.0])
        B1 = sparse.csc_matrix((vals, (rows, cols)), shape=(n_v, n_e))
    else:
        B1 = sparse.csc_matrix((n_v, 0))

    # B2: (n_e x n_t)
    if n_t > 0 and n_e > 0:
        rows, cols, vals = [], [], []
        for t_idx, (i, j, k) in enumerate(triangles):
            # Boundary of (i,j,k) = (j,k) - (i,k) + (i,j)
            # with the convention that edges are sorted (smaller, larger)
            e_ij = e_idx.get((i, j))
            e_ik = e_idx.get((i, k))
            e_jk = e_idx.get((j, k))
            if e_ij is not None and e_ik is not None and e_jk is not None:
                rows.extend([e_jk, e_ik, e_ij])
                cols.extend([t_idx, t_idx, t_idx])
                vals.extend([1.0, -1.0, 1.0])
        B2 = sparse.csc_matrix((vals, (rows, cols)), shape=(n_e, n_t))
    else:
        B2 = sparse.csc_matrix((n_e, 0))

    return B1, B2


# ---------------------------------------------------------------------------
# Hodge Laplacians and spectral analysis
# ---------------------------------------------------------------------------

def hodge_analysis(B1, B2):
    """Compute Hodge Laplacians and Betti numbers.

    L0 = B1 B1^T                (vertex Laplacian)
    L1 = B1^T B1 + B2 B2^T      (edge Laplacian)

    Args:
        B1: sparse (n_v, n_e) boundary operator.
        B2: sparse (n_e, n_t) boundary operator.

    Returns:
        dict with keys: L0, L1, eigenvalues_L0, eigenvalues_L1,
                        beta_0, beta_1, n_vertices, n_edges, n_triangles.
    """
    n_v, n_e = B1.shape
    _, n_t = B2.shape

    # L0 = B1 @ B1.T
    L0 = (B1 @ B1.T).toarray().astype(np.float64)

    # L1 = B1.T @ B1 + B2 @ B2.T
    if n_e > 0:
        L1_down = (B1.T @ B1).toarray().astype(np.float64)
        L1_up = (B2 @ B2.T).toarray().astype(np.float64)
        L1 = L1_down + L1_up
    else:
        L1 = np.zeros((0, 0))

    # Eigenvalues
    eigs_L0 = np.sort(np.linalg.eigvalsh(L0)) if L0.size > 0 else np.array([])
    eigs_L1 = np.sort(np.linalg.eigvalsh(L1)) if L1.size > 0 else np.array([])

    # Betti numbers: count near-zero eigenvalues
    tol = 1e-8
    beta_0 = int(np.sum(eigs_L0 < tol)) if len(eigs_L0) > 0 else n_v
    beta_1 = int(np.sum(eigs_L1 < tol)) if len(eigs_L1) > 0 else 0

    return {
        "L0": L0,
        "L1": L1,
        "eigenvalues_L0": eigs_L0,
        "eigenvalues_L1": eigs_L1,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "n_vertices": n_v,
        "n_edges": n_e,
        "n_triangles": n_t,
    }


# ---------------------------------------------------------------------------
# Hodge decomposition of a 1-cochain
# ---------------------------------------------------------------------------

def hodge_decomposition(B1, B2, f):
    """Decompose a 1-cochain f into gradient + curl + harmonic components.

    f = B1^T g + B2 c + h

    where:
        - gradient component: B1^T g  (exact / in image of B1^T)
        - curl component:     B2 c    (coexact / in image of B2)
        - harmonic component: h       (in kernel of L1)

    Uses least-squares projections.

    Args:
        B1: sparse (n_v, n_e)
        B2: sparse (n_e, n_t)
        f:  1-cochain vector of length n_e

    Returns:
        dict with keys: gradient, curl, harmonic, and their norms.
    """
    n_e = len(f)
    f = np.asarray(f, dtype=np.float64)

    # Gradient component: project f onto image(B1^T)
    # Solve B1 B1^T g = B1 f  =>  g = pinv(L0_down) B1 f
    # Then gradient = B1^T g
    B1_dense = B1.toarray().astype(np.float64)
    B1T = B1_dense.T

    L0 = B1_dense @ B1T  # Not the graph Laplacian, but B1 B1^T acting on cochains
    # Actually: gradient = B1^T @ pinv(B1 @ B1^T) @ B1 @ f
    # Use pseudoinverse for robustness
    if L0.size > 0 and n_e > 0:
        # Project onto image(B1^T):  P_grad = B1^T (B1 B1^T)^+ B1
        L0_pinv = np.linalg.pinv(L0)
        gradient = B1T @ L0_pinv @ B1_dense @ f
    else:
        gradient = np.zeros(n_e)

    # Curl component: project f onto image(B2)
    # curl = B2 @ pinv(B2^T B2) @ B2^T @ f
    B2_dense = B2.toarray().astype(np.float64)
    if B2_dense.shape[1] > 0 and n_e > 0:
        L_up = B2_dense.T @ B2_dense
        L_up_pinv = np.linalg.pinv(L_up)
        curl = B2_dense @ L_up_pinv @ B2_dense.T @ f
    else:
        curl = np.zeros(n_e)

    # Harmonic component: whatever is left
    harmonic = f - gradient - curl

    return {
        "gradient": gradient,
        "curl": curl,
        "harmonic": harmonic,
        "norm_gradient": float(np.linalg.norm(gradient)),
        "norm_curl": float(np.linalg.norm(curl)),
        "norm_harmonic": float(np.linalg.norm(harmonic)),
        "norm_f": float(np.linalg.norm(f)),
    }


# ---------------------------------------------------------------------------
# Filtration analysis
# ---------------------------------------------------------------------------

def filtration_analysis(barrier_matrix, triplet_barriers, tau_range):
    """Sweep tau and record topological invariants at each threshold.

    Args:
        barrier_matrix: (n, n) pairwise barrier matrix.
        triplet_barriers: dict mapping (i,j,k) -> barrier value.
        tau_range: iterable of tau values (should be sorted ascending).

    Returns:
        list of dicts, each with keys: tau, beta_0, beta_1, n_edges,
        n_triangles, fill_ratio.
    """
    n = barrier_matrix.shape[0]
    max_edges = n * (n - 1) // 2

    results = []
    for tau in tau_range:
        vertices, edges, triangles = build_mergeability_complex(
            barrier_matrix, triplet_barriers, tau
        )
        B1, B2 = build_boundary_operators(vertices, edges, triangles)
        analysis = hodge_analysis(B1, B2)

        # fill_ratio = triangles / 3-cliques (fraction of 3-cliques that are filled)
        edge_set = set(edges)
        n_3cliques = sum(
            1 for triple in combinations(range(n), 3)
            if (triple[0], triple[1]) in edge_set
            and (triple[0], triple[2]) in edge_set
            and (triple[1], triple[2]) in edge_set
        )
        fill_ratio = len(triangles) / n_3cliques if n_3cliques > 0 else 1.0

        results.append({
            "tau": float(tau),
            "beta_0": analysis["beta_0"],
            "beta_1": analysis["beta_1"],
            "n_edges": len(edges),
            "n_triangles": len(triangles),
            "fill_ratio": fill_ratio,
            "eigenvalues_L1": analysis["eigenvalues_L1"].tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hodge analysis on mergeability complex")
    parser.add_argument("--barriers", type=str, required=True,
                        help="Path to barriers .npz file")
    parser.add_argument("--output", type=str, default="hodge_results.json",
                        help="Output JSON file path")
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=0.2)
    parser.add_argument("--tau-steps", type=int, default=50)
    args = parser.parse_args()

    # Load barrier data
    data = np.load(args.barriers, allow_pickle=True)
    barrier_matrix = data["barrier_matrix"]
    triplet_keys = data["triplet_keys"]
    triplet_vals = data["triplet_vals"]

    # Reconstruct triplet dict
    triplet_barriers = {}
    if len(triplet_keys) > 0:
        for key, val in zip(triplet_keys, triplet_vals):
            triplet_barriers[tuple(key)] = float(val)

    n = barrier_matrix.shape[0]
    print(f"Loaded barrier data: {n} models")
    print(f"  Pairwise barrier range: [{barrier_matrix[barrier_matrix > 0].min():.4f}, "
          f"{barrier_matrix.max():.4f}]" if barrier_matrix.max() > 0 else
          "  All barriers are zero (identical models?)")
    print(f"  Triplet barriers: {len(triplet_barriers)}")

    # Filtration
    tau_range = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
    print(f"\nRunning filtration over tau in [{args.tau_min}, {args.tau_max}] "
          f"({args.tau_steps} steps) ...")
    results = filtration_analysis(barrier_matrix, triplet_barriers, tau_range)

    # Print summary
    print(f"\n{'tau':>8s} {'beta_0':>6s} {'beta_1':>6s} {'edges':>6s} "
          f"{'tri':>6s} {'fill':>6s}")
    print("-" * 44)
    for r in results:
        print(f"{r['tau']:8.4f} {r['beta_0']:6d} {r['beta_1']:6d} "
              f"{r['n_edges']:6d} {r['n_triangles']:6d} {r['fill_ratio']:6.3f}")

    # Hodge decomposition at the median tau that has edges
    taus_with_edges = [r for r in results if r["n_edges"] > 0]
    if taus_with_edges:
        mid = taus_with_edges[len(taus_with_edges) // 2]
        tau_mid = mid["tau"]
        print(f"\n--- Hodge decomposition at tau = {tau_mid:.4f} ---")
        vertices, edges, triangles = build_mergeability_complex(
            barrier_matrix, triplet_barriers, tau_mid
        )
        B1, B2 = build_boundary_operators(vertices, edges, triangles)

        # Use barrier values on edges as a 1-cochain
        f = np.array([barrier_matrix[e[0], e[1]] for e in edges])
        decomp = hodge_decomposition(B1, B2, f)
        print(f"  ||f||       = {decomp['norm_f']:.4f}")
        print(f"  ||gradient||= {decomp['norm_gradient']:.4f}")
        print(f"  ||curl||    = {decomp['norm_curl']:.4f}")
        print(f"  ||harmonic||= {decomp['norm_harmonic']:.4f}")

        # Add decomposition to output
        decomp_out = {k: v for k, v in decomp.items()
                      if isinstance(v, float)}
        decomp_out["tau"] = tau_mid
    else:
        decomp_out = None

    # Save
    output = {
        "filtration": results,
        "hodge_decomposition": decomp_out,
        "n_models": n,
        "model_names": data["model_names"].tolist() if "model_names" in data else [],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
