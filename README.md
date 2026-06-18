# Hodge-Guided Model Alignment

Code for *"Hodge-Guided Model Alignment: Decomposing and Reducing Topological Merge Obstructions"*.

## Result

Standard star-topology alignment accumulates 25.5% harmonic residual (irreducible topological obstruction). MST alignment along minimum-harmonic paths reduces it by 93% (25.5% → 1.8%), collapsing β₁ from 75 to 5.

## Paper Correspondence

| Paper Section | Script / Module | What it produces |
|---|---|---|
| §2.1 Mergeability complex K_τ | `src/topology/hodge.py` | Filtration, Betti numbers, fill ratio |
| §2.2 Hodge decomposition | `src/topology/hodge.py` | Gradient / curl / harmonic split |
| §2.3 Hodge-guided MST alignment | `run_topo_alignment.py` | MST barriers, energy redistribution |
| §3.1 24 PlainCNN models | `src/zoo/plain_cnn.py` | Trained checkpoints in `checkpoints_plain/` |
| §3.2 Complex structure (Fig 1) | `run_full_pipeline.py` | β₁ peak at 122, fill ratio curve |
| §3.3 Hodge decomposition & node potential (Fig 2, Table 1) | `run_hodge_attribution.py` | Per-edge decomposition, merge potential s |
| §3.4 Anomaly detection (Fig 3) | `run_anomaly_detection.py` | Per-model harmonic fraction profile |
| §3.5 Star vs MST (Fig 4, Table 2) | `run_topo_alignment.py` | Side-by-side barrier comparison |
| All figures | `generate_paper_figures.py` | Fig 1–4 |

## Repository Structure

```
src/
  topology/hodge.py       # Mergeability complex, Hodge Laplacian, decomposition
  barriers/align.py       # Weight-matching alignment (Git Re-Basin style)
  zoo/plain_cnn.py        # PlainCNN model definition and training
run_full_pipeline.py      # Step 1: Align → barriers → Hodge → filtration
run_topo_alignment.py     # Step 2: Star vs MST alignment comparison
run_hodge_attribution.py  # Step 3: Node potential and per-edge decomposition
run_anomaly_detection.py  # Step 4: Per-model harmonic anomaly profile
generate_paper_figures.py # Step 5: Generate all paper figures
checkpoints_plain/        # 24 pretrained PlainCNN models (8 seeds × 3 LRs)
results_24/               # Star-alignment barrier data and filtration
results/                  # MST alignment, Hodge attribution, anomaly detection
```

## Reproduction

### Prerequisites

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib
```

### Run pipeline (order matters)

```bash
# Step 1: Full pipeline — align, compute barriers, Hodge analysis (~2h on 1 GPU)
python run_full_pipeline.py --ckpt-dir checkpoints_plain --output-dir results_24 --device cuda:0

# Step 2: MST alignment comparison (~1h on 1 GPU, reuses Step 1 data)
python run_topo_alignment.py --device cuda:0

# Step 3: Hodge attribution analysis (seconds, CPU only)
python run_hodge_attribution.py

# Step 4: Anomaly detection (seconds, CPU only)
python run_anomaly_detection.py

# Step 5: Generate figures
python generate_paper_figures.py
```

Steps 3–5 only require the output of Steps 1–2 and run in seconds.

### Pre-computed results

All result files are included. To regenerate figures without rerunning experiments:

```bash
python generate_paper_figures.py
```

## Data

- **24 models**: PlainCNN (~1.2M params), CIFAR-10, 8 seeds × 3 learning rates (0.01, 0.05, 0.1), 100 epochs, SGD + cosine annealing
- **276 pairwise barriers** + **884 triplet barriers** (star alignment)
- **276 pairwise barriers** + **899 triplet barriers** (MST alignment)
