"""Ablate the two-level week x activity-type PCG against its controls.

Writes to ``results/graph_ablation/`` so the existing ``results/full_evaluation/``
outputs stay untouched and comparable.

    python scripts/run_graph_ablation.py --dataset oulab --folds 5 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.full_evaluation import _bootstrap_mean_ci, run_dataset_evaluation  # noqa: E402

MODELS = (
    "temporal_hgb",             # the bar the path-graph PCG never cleared
    "pcg_ut",                   # the original path-graph PCG
    # -- nested information ladder (non-destructive) --------------------
    "pcg_g_local_only",         # nodes in isolation: no cross-modality view
    "pcg_g_no_graph",           # + structural summaries, still no propagation
    "pcg_g_residual",           # + graph-derived residuals (strict superset)
    "pcg_g_residual_shuffled",  # residuals against the WRONG modality history
    # -- destructive propagation (overwrites node states) ---------------
    "pcg_g",                    # two-level graph, both channels
    "pcg_g_shuffled",           # structure destroyed, marginals preserved
    "pcg_g_temporal_only",      # (w,a) -> (w+1,a) only
    "pcg_g_hierarchy_only",     # (w,*) <-> W_w only
)

#: Ordered low -> high information content; AUC should rise along this if graph
#: structure carries signal, since each step strictly adds feature columns.
LADDER = ("pcg_g_local_only", "pcg_g_no_graph", "pcg_g_residual")

WEEKS = {"oulab": [2, 4, 6, 8, 10, 12, 14, 16], "kdd": [1, 2, 3, 4, 5]}


def paired_deltas(folds: pd.DataFrame, reference: str = "pcg_g") -> pd.DataFrame:
    """Per-fold AUC differences against ``reference`` on identical splits."""
    join = ["dataset", "week", "repeat", "fold"]
    ref = folds[folds["model"] == reference][join + ["auc"]]
    rows = []
    for model in folds["model"].unique():
        if model == reference:
            continue
        other = folds[folds["model"] == model][join + ["auc"]]
        paired = ref.merge(other, on=join, suffixes=("_ref", "_other"))
        for week, group in paired.groupby("week"):
            delta = (group["auc_ref"] - group["auc_other"]).to_numpy()
            mean, low, high = _bootstrap_mean_ci(delta)
            rows.append({
                "week": week,
                "comparison": f"{reference} - {model}",
                "n_paired_folds": len(delta),
                "auc_delta_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "separates": bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows).sort_values(["week", "comparison"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oulab", choices=("oulab", "kdd"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = Path("results") / "graph_ablation"
    out.mkdir(parents=True, exist_ok=True)

    folds, _ = run_dataset_evaluation(
        dataset=args.dataset,
        snapshot_weeks=WEEKS[args.dataset],
        n_splits=args.folds,
        repeats=args.repeats,
        output_root=out,
        verbose=args.verbose,
        models=MODELS,
    )

    summary = (
        folds.groupby(["dataset", "week", "model"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "auc_mean", "std": "auc_std", "count": "n_folds"})
    )
    summary.to_csv(out / f"{args.dataset}_summary.csv", index=False)

    deltas = paired_deltas(folds, reference="pcg_g_residual")
    deltas.to_csv(out / f"{args.dataset}_paired_deltas.csv", index=False)

    pivot = summary.pivot(index="week", columns="model", values="auc_mean")
    print("\n=== AUC by week ===")
    print(pivot.round(4).to_string())

    print("\n=== nested information ladder (each step strictly adds columns) ===")
    ladder = [m for m in LADDER if m in pivot.columns]
    print(pivot[ladder].round(4).to_string())
    rising = (pivot[ladder].diff(axis=1).iloc[:, 1:] > 0).all(axis=1)
    print(f"\nweeks where AUC rises monotonically along the ladder: "
          f"{int(rising.sum())}/{len(rising)}")

    print("\n=== pcg_g_residual vs controls (paired, per fold) ===")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
