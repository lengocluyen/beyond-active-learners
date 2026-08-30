"""Cluster-bootstrap interval for the AGGREGATE decomposition components.

Table VI reports medians and identity-preserving means across 88
model--landmark cells, plus the share of cellwise intervals excluding zero.
Those shares are descriptive and are not significance tests, so the headline
claim -- that the evaluation component dominates the training component --
still lacks a direct interval.

This script supplies one. Within every bootstrap draw the same presentation
cluster weights are applied to all cells, each component is recomputed, and the
mean across cells is taken. The percentile interval of that aggregate is a
direct statement about the quantity the abstract claims.

Aggregating means rather than medians is deliberate: the symmetric attribution
is an identity within a cell, and the mean is the only summary that preserves
it, so the aggregate evaluation and training components still sum to the
aggregate joint difference inside every draw.

Usage
-----
    python scripts/aggregate_decomposition_ci.py \
        --results results/cohort_exchange_2x2_clean/oulab --n-boot 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort_exchange import (
    CROSS_PROTOCOL,
    CROSS_PROTOCOL_VA,
    METRICS,
    _bootstrap_cluster_counts,
    _weighted_cluster_metric_draws,
)

AA, VV = "activity_conditioned", "cutoff_valid"
AV, VA = CROSS_PROTOCOL, CROSS_PROTOCOL_VA

COMPONENTS = {
    "joint": {AA: 1.0, VV: -1.0},
    "evaluation (symmetric)": {AA: 0.5, AV: -0.5, VA: 0.5, VV: -0.5},
    "training (symmetric)": {AA: 0.5, VA: -0.5, AV: 0.5, VV: -0.5},
}


def aggregate_draws(results: Path, n_boot: int, seed: int) -> dict:
    cols = ["week", "model", "protocol", "y", "p_success", "cluster_id"]
    preds = pd.read_csv(results / "predictions.csv.gz", usecols=cols)
    required = {AA, AV, VA, VV}

    # metric -> component -> list of per-cell draw arrays
    stacks: dict[str, dict[str, list[np.ndarray]]] = {
        m: {c: [] for c in COMPONENTS} for m in METRICS
    }
    n_cells = 0
    for (week, model), cell in preds.groupby(["week", "model"]):
        if not required.issubset(set(cell.protocol)):
            continue
        clusters = cell["cluster_id"].dropna().astype(str).unique()
        # Reseeding per cell reproduces the identical cluster weights in every
        # cell, which is what makes averaging across cells within a draw valid.
        rng = np.random.default_rng(seed)
        counts = _bootstrap_cluster_counts(n_boot, len(clusters), rng)
        draws = {
            p: _weighted_cluster_metric_draws(
                cell[cell.protocol.eq(p)], counts, clusters
            )
            for p in required
        }
        for metric in METRICS:
            for name, weights in COMPONENTS.items():
                total = np.zeros(n_boot)
                for protocol, coef in weights.items():
                    total = total + coef * draws[protocol][metric]
                stacks[metric][name].append(total)
        n_cells += 1
        print(f"  cell {n_cells:>3}: week={week} model={model}", flush=True)
    return stacks, n_cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("results/cohort_exchange_2x2_clean/oulab"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stacks, n_cells = aggregate_draws(args.results, args.n_boot, args.seed)
    print(f"\nAggregate mean components over {n_cells} model-landmark cells, "
          f"{args.n_boot} paired presentation-cluster draws\n")
    print(f"{'metric':<14}{'component':<24}{'mean':>10}{'95% CI':>22}")
    for metric in METRICS:
        for name in COMPONENTS:
            per_cell = np.vstack(stacks[metric][name])       # cells x draws
            agg = np.nanmean(per_cell, axis=0)               # draws
            agg = agg[np.isfinite(agg)]
            lo, hi = np.quantile(agg, [0.025, 0.975])
            print(f"{metric:<14}{name:<24}{agg.mean():>+10.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")
        print()


if __name__ == "__main__":
    main()
