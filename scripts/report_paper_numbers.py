"""Print every aggregate number the IEEE TLT manuscript quotes.

The manuscript must never be edited from memory: run this against a completed
results directory and transcribe. All four cells of the 2x2 design have to come
from the *same* run, otherwise the training-population contrast compares two
code versions rather than two populations.

Usage
-----
    python scripts/report_paper_numbers.py \
        --results results/cohort_exchange_2x2_clean/oulab

Add ``--kdd results/cohort_exchange_references/kdd`` for the second-dataset
numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_LABELS = {
    "auc": "ROC-AUC",
    "pr_auc_risk": "adverse-outcome PR-AUC",
    "brier": "Brier score",
    "ece": "ECE",
}

CONTRASTS = [
    ("joint_population_difference", "joint  AA-VV"),
    ("evaluation_population_shapley", "SYMMETRIC evaluation"),
    ("training_population_shapley", "SYMMETRIC training"),
    ("evaluation_population_component", "  path1 eval   AA-AV"),
    ("evaluation_population_component_at_valid", "  path2 eval   VA-VV"),
    ("training_population_component", "  path1 train  AV-VV"),
    ("training_population_component_at_activity", "  path2 train  AA-VA"),
]


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def decomposition(results: Path) -> None:
    path = results / "training_evaluation_decomposition.csv"
    if not path.exists():
        print(f"  MISSING {path.name} -- run the benchmark first")
        return
    d = pd.read_csv(path)
    _rule("RQ2  training-by-evaluation decomposition  (Table: tab:decomposition)")
    residual = d["identity_residual"].abs().max()
    print(f"identity residual (must be ~0): {residual:.3e}")
    per_contrast = d.groupby("contrast").size().unique()
    print(f"cells per contrast: {per_contrast} | models={d.model.nunique()} "
          f"| weeks={d.week.nunique()}")
    for metric, label in METRIC_LABELS.items():
        frame = d[d.metric.eq(metric)]
        if frame.empty:
            continue
        print(f"\n--- {label} ---")
        for contrast, short in CONTRASTS:
            g = frame[frame.contrast.eq(contrast)]
            if g.empty:
                print(f"{short:<24} (absent -- V->A cell not in this run)")
                continue
            excl = ((g.ci_low > 0) | (g.ci_high < 0)).mean() * 100
            print(f"{short:<24} median={g.delta.median():+.4f}"
                  f"   CI excludes 0: {excl:5.1f}%")
        p1 = frame[frame.contrast.eq("evaluation_population_component")]
        p2 = frame[frame.contrast.eq("evaluation_population_component_at_valid")]
        if not p2.empty:
            key = ["week", "model"]
            gap = p1.set_index(key).delta - p2.set_index(key).delta
            print(f"{'  interaction (p1-p2)':<24} median={gap.median():+.4f}"
                  f"   IQR=[{gap.quantile(.25):+.4f},{gap.quantile(.75):+.4f}]")


def population(results: Path) -> None:
    path = results / "cohort_composition.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    idx = c[c.table.eq("indices")].sort_values("week")
    _rule("RQ1  population exchange  (Table: tab:composition)")
    cols = ["week", "n_activity", "n_cutoff_valid", "known_outcomes_in_activity",
            "eligible_silent_excluded", "cohort_jaccard", "risk_concentration_ratio"]
    print(idx[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.3f}"))


def decisions(results: Path, budget: float = 0.05) -> None:
    path = results / "budget_performance.csv"
    if not path.exists():
        return
    b = pd.read_csv(path)
    b = b[b.budget.eq(budget)]
    _rule(f"RQ3  fixed-budget decisions at {budget:.0%}")
    for protocol, g in b.groupby("protocol"):
        print(f"  {protocol:<26} noneligible allocation={g.wasted_budget_rate.median():.3f}"
              f"   eligibility-adjusted precision={g.actionable_precision.median():.3f}")
    overlap = results / "decision_overlap.csv"
    if overlap.exists():
        o = pd.read_csv(overlap)
        o = o[o.budget.eq(budget) & o.week_from.eq(o.week_to)]
        print(f"\n  top-{budget:.0%} list agreement (median Jaccard):")
        for (p, r), g in o.groupby(["protocol", "reference_protocol"]):
            if p == r or len(g) < 20:
                continue
            print(f"    {p[:26]:<27} vs {r[:26]:<27} {g.jaccard.median():.3f}")


def invariant(results: Path) -> None:
    """Shared learners must score identically within a row of the design."""
    path = results / "predictions.csv.gz"
    if not path.exists():
        print("\n  (predictions.csv.gz absent -- skipping invariant check)")
        return
    _rule("INVARIANT  shared learners score identically within each design row")
    key = ["code_module", "code_presentation", "id_student"]
    cols = ["week", "model", "protocol", "p_success", *key]
    d = pd.read_csv(path, usecols=cols)
    pairs = [("activity_conditioned", "activity_to_cutoff_valid"),
             ("cutoff_valid", "cutoff_valid_to_activity")]
    worst = 0.0
    for left, right in pairs:
        for (week, model), cell in d.groupby(["week", "model"]):
            a = cell[cell.protocol.eq(left)][key + ["p_success"]]
            b = cell[cell.protocol.eq(right)][key + ["p_success"]]
            if a.empty or b.empty:
                continue
            m = a.merge(b, on=key, suffixes=("_l", "_r"))
            if m.empty:
                continue
            worst = max(worst, float((m.p_success_l - m.p_success_r).abs().max()))
    print(f"  max |score difference| over all shared learners: {worst:.3e}")
    print("  (must be exactly 0.0 -- any drift means features moved with the protocol)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("results/cohort_exchange_2x2_clean/oulab"))
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--skip-invariant", action="store_true",
                        help="the invariant check streams the full prediction file")
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"results directory not found: {args.results}")
    print(f"reporting from: {args.results}")
    population(args.results)
    decomposition(args.results)
    decisions(args.results, args.budget)
    if not args.skip_invariant:
        invariant(args.results)


if __name__ == "__main__":
    main()
