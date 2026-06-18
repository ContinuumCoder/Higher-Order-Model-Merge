#!/usr/bin/env python3
"""Generate publication-quality figures for the Mergeability Complex workshop paper."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

OUT = Path("WSL")
OUT.mkdir(exist_ok=True)

# Color palette (colorblind-friendly)
SEED_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))
LR_MARKERS = {0.01: "o", 0.05: "s", 0.1: "^"}


def shorten(name: str) -> str:
    """plaincnn_s0_lr0.01_wd0.0001_augbasic.pt -> s0_lr0.01"""
    name = name.replace("plaincnn_", "").replace("_wd0.0001_augbasic.pt", "")
    return name


# ===== Load data =====
with open("results_24/report.json") as f:
    report = json.load(f)
filt = report["filtration"]

bsm = np.load("results/barriers_star_vs_mst.npz", allow_pickle=True)
barrier_star = bsm["barrier_star"]
barrier_mst = bsm["barrier_mst"]
model_names_sm = [shorten(n) for n in bsm["names"]]

with open("results/topo_alignment.json") as f:
    topo = json.load(f)

with open("results/hodge_attribution.json") as f:
    hodge = json.load(f)

with open("results/anomaly_detection.json") as f:
    anomaly = json.load(f)


# ===================================================================
# Figure 1: Filtration (2 stacked subplots)
# ===================================================================
def fig1_filtration():
    taus = [d["tau"] for d in filt]
    b0 = [d["beta_0"] for d in filt]
    b1 = [d["beta_1"] for d in filt]
    fr = [d["fill_ratio"] for d in filt]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(taus, b0, color="C0", lw=1.5, label=r"$\beta_0$")
    ax1.plot(taus, b1, color="C3", lw=1.5, label=r"$\beta_1$")

    # Annotate peak beta_1
    peak_idx = int(np.argmax(b1))
    ax1.annotate(
        rf"$\beta_1 = {b1[peak_idx]}$",
        xy=(taus[peak_idx], b1[peak_idx]),
        xytext=(taus[peak_idx] + 0.15, b1[peak_idx] + 8),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="C3", lw=0.8),
        color="C3",
    )

    ax1.set_ylabel("Betti number")
    ax1.legend(frameon=False)

    ax2.plot(taus, fr, color="C2", lw=1.5)
    ax2.set_xlabel(r"Threshold $\tau$")
    ax2.set_ylabel("Fill ratio")

    fig.align_ylabels()
    plt.tight_layout(h_pad=0.3)
    fig.savefig(OUT / "fig1_filtration.pdf")
    print("  Saved fig1_filtration.pdf")
    plt.close(fig)


# ===================================================================
# Figure 2: Star vs MST (2x2)
# ===================================================================
def fig2_star_vs_mst():
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), constrained_layout=True)

    vmin = 0
    vmax = max(barrier_star.max(), barrier_mst.max())

    # (a) Star barrier heatmap
    ax = axes[0, 0]
    im = ax.imshow(barrier_star, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(model_names_sm)))
    ax.set_yticks(range(len(model_names_sm)))
    ax.set_xticklabels(model_names_sm, rotation=90, fontsize=5)
    ax.set_yticklabels(model_names_sm, fontsize=5)
    ax.set_title("(a) Star graph barriers", fontsize=9)

    # (b) MST barrier heatmap
    ax = axes[0, 1]
    im2 = ax.imshow(barrier_mst, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(model_names_sm)))
    ax.set_yticks(range(len(model_names_sm)))
    ax.set_xticklabels(model_names_sm, rotation=90, fontsize=5)
    ax.set_yticklabels(model_names_sm, fontsize=5)
    ax.set_title("(b) MST graph barriers", fontsize=9)

    # Shared colorbar for top row
    fig.colorbar(im2, ax=axes[0, :], shrink=0.6, label="Barrier", pad=0.02)

    # (c) Hodge energy comparison bar chart
    ax = axes[1, 0]
    categories = ["Gradient", "Curl", "Harmonic"]
    star_vals = [topo["star"]["grad_pct"], topo["star"]["curl_pct"], topo["star"]["harm_pct"]]
    mst_vals = [topo["mst"]["grad_pct"], topo["mst"]["curl_pct"], topo["mst"]["harm_pct"]]

    x = np.arange(len(categories))
    w = 0.3
    bars1 = ax.bar(x - w / 2, star_vals, w, label="Star", color="C0", edgecolor="k", linewidth=0.4)
    bars2 = ax.bar(x + w / 2, mst_vals, w, label="MST", color="C1", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Energy (%)")
    ax.set_title("(c) Hodge decomposition", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=6)

    # (d) beta_1 comparison
    ax = axes[1, 1]
    b1_star = topo["star"]["beta_1"]
    b1_mst = topo["mst"]["beta_1"]
    bars = ax.bar(["Star", "MST"], [b1_star, b1_mst],
                  color=["C0", "C1"], edgecolor="k", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, [b1_star, b1_mst]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"$\beta_1$")
    ax.set_title(r"(d) Loop complexity $\beta_1$", fontsize=9)

    fig.savefig(OUT / "fig2_star_vs_mst.pdf")
    print("  Saved fig2_star_vs_mst.pdf")
    plt.close(fig)


# ===================================================================
# Figure 3: Anomaly Detection Scatter
# ===================================================================
def fig3_anomaly():
    stats = anomaly["model_stats"]

    fig, ax = plt.subplots(figsize=(5, 4))

    seeds = sorted(set(s["seed"] for s in stats))
    lrs = sorted(set(s["lr"] for s in stats))
    seed_cmap = {s: SEED_COLORS[i] for i, s in enumerate(seeds)}

    barriers = [s["mean_barrier"] for s in stats]
    harms = [s["mean_harm_frac"] for s in stats]

    for s in stats:
        ax.scatter(
            s["mean_barrier"], s["mean_harm_frac"],
            c=[seed_cmap[s["seed"]]],
            marker=LR_MARKERS[s["lr"]],
            s=50, edgecolors="k", linewidths=0.4, zorder=3,
        )

    # Median lines
    med_b = np.median(barriers)
    med_h = np.median(harms)
    ax.axvline(med_b, ls="--", color="grey", lw=0.8, zorder=1)
    ax.axhline(med_h, ls="--", color="grey", lw=0.8, zorder=1)

    # Quadrant labels
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    pad_x = 0.01 * (xhi - xlo)
    pad_y = 0.01 * (yhi - ylo)
    ax.text(xlo + pad_x, ylo + pad_y, "Compatible", fontsize=7,
            color="green", fontstyle="italic", va="bottom", ha="left")
    ax.text(xhi - pad_x, ylo + pad_y, "Fixable\noutlier", fontsize=7,
            color="orange", fontstyle="italic", va="bottom", ha="right")
    ax.text(xhi - pad_x, yhi - pad_y, "Toxic\noutlier", fontsize=7,
            color="red", fontstyle="italic", va="top", ha="right")
    ax.text(xlo + pad_x, yhi - pad_y, "Surprising", fontsize=7,
            color="C4", fontstyle="italic", va="top", ha="left")

    ax.set_xlabel("Mean pairwise barrier")
    ax.set_ylabel("Mean harmonic fraction")

    # Legends
    seed_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=seed_cmap[s], markeredgecolor="k",
                           markersize=6, label=f"seed {s}") for s in seeds]
    lr_handles = [Line2D([0], [0], marker=LR_MARKERS[lr], color="w",
                         markerfacecolor="grey", markeredgecolor="k",
                         markersize=6, label=f"lr={lr}") for lr in lrs]
    leg1 = ax.legend(handles=seed_handles, loc="upper left", frameon=False,
                     fontsize=6, ncol=2, title="Seed", title_fontsize=7,
                     bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=lr_handles, loc="lower right", frameon=False,
              fontsize=7, title="LR", title_fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT / "fig3_anomaly.pdf")
    print("  Saved fig3_anomaly.pdf")
    plt.close(fig)


# ===================================================================
# Figure 4: Node Potential vs Seed
# ===================================================================
def fig4_node_potential():
    # Use tau with most edges
    best_key = max(hodge.keys(), key=lambda k: hodge[k]["n_edges"])
    hdata = hodge[best_key]
    node_pot = hdata["node_potential"]  # dict: model_name -> value
    rho_seed = hdata["correlation_s_seed"]["rho"]
    p_seed = hdata["correlation_s_seed"]["p"]

    # Parse seed from model name
    names = list(node_pot.keys())
    potentials = np.array([node_pot[n] for n in names])
    seeds = np.array([int(n.split("_")[0][1:]) for n in names])

    # Sort by potential
    order = np.argsort(potentials)
    names_s = [names[i] for i in order]
    pot_s = potentials[order]
    seeds_s = seeds[order]

    unique_seeds = sorted(set(seeds))
    seed_cmap = {s: SEED_COLORS[i] for i, s in enumerate(unique_seeds)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2),
                                     gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: node potential bar
    colors = [seed_cmap[s] for s in seeds_s]
    ax1.bar(range(len(pot_s)), pot_s, color=colors, edgecolor="k", linewidth=0.3)
    ax1.set_xlabel("Model index (sorted by node potential $s$)")
    ax1.set_ylabel("Node potential $s$")
    ax1.set_xticks(range(len(pot_s)))
    ax1.set_xticklabels([shorten_short(n) for n in names_s], rotation=90, fontsize=5)

    handles = [Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=seed_cmap[s], markeredgecolor="k",
                       markersize=6, label=f"seed {s}") for s in unique_seeds]
    ax1.legend(handles=handles, frameon=False, fontsize=6, ncol=2,
               loc="upper left", title="Seed", title_fontsize=7)

    # Right panel: correlation annotation
    ax2.axis("off")
    ax2.text(0.5, 0.55, rf"$\rho_{{s,\mathrm{{seed}}}} = {rho_seed:.3f}$",
             transform=ax2.transAxes, fontsize=14, ha="center", va="center",
             fontweight="bold")
    ax2.text(0.5, 0.40, rf"$p = {p_seed:.1e}$",
             transform=ax2.transAxes, fontsize=10, ha="center", va="center",
             color="grey")
    ax2.text(0.5, 0.25, "Spearman rank\ncorrelation",
             transform=ax2.transAxes, fontsize=8, ha="center", va="center",
             color="grey")

    plt.tight_layout()
    fig.savefig(OUT / "fig4_node_potential.pdf")
    print("  Saved fig4_node_potential.pdf")
    plt.close(fig)


def shorten_short(name):
    """s0_lr0.01 -> s0/.01"""
    parts = name.split("_")
    seed = parts[0]
    lr = parts[1].replace("lr", "")
    return f"{seed}/{lr}"


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_filtration()
    fig2_star_vs_mst()
    fig3_anomaly()
    fig4_node_potential()
    print("Done.")
