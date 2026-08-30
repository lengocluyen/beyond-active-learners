"""Active-and-eligible comparator: C_act(t) intersect C_valid(t).

This isolates the two mechanisms that the activity-conditioned protocol
combines. The intersection keeps the event-first candidate rule but removes
learners whose withdrawal is already recorded, so any remaining loss relative
to the cutoff-valid protocol is attributable to omitting eligible silent
learners rather than to retaining resolved outcomes.

No refitting is required: the activity-trained out-of-fold scores are simply
restricted to the intersection, while intervention capacity stays fixed at the
cutoff-valid population so that protocols cannot gain by shrinking their own
candidate pool.

Usage
-----
    python scripts/active_eligible_comparator.py \
        --results results/cohort_exchange_2x2_clean/oulab --budget 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
PRESENTATION = ["code_module", "code_presentation"]


def _capacity(valid: pd.DataFrame, budget: float) -> dict[tuple, int]:
    """Capacity is always defined on the cutoff-valid risk set."""
    sizes = valid.groupby(PRESENTATION).size()
    return {k: max(1, int(round(budget * n))) for k, n in sizes.items()}


def _select(frame: pd.DataFrame, capacity: dict[tuple, int]) -> pd.DataFrame:
    picks = []
    for key, group in frame.groupby(PRESENTATION):
        k = min(len(group), capacity.get(key, 0))
        if k > 0:
            picks.append(group.nlargest(k, "risk_score"))
    return pd.concat(picks, ignore_index=True) if picks else frame.iloc[:0]


def evaluate(results: Path, budget: float) -> pd.DataFrame:
    cols = ["week", "model", "protocol", "risk", "cutoff_valid",
            "membership_class", "risk_score", *KEY]
    preds = pd.read_csv(results / "predictions.csv.gz", usecols=cols)
    preds["cutoff_valid"] = preds["cutoff_valid"].astype(bool)

    rows = []
    for (week, model), cell in preds.groupby(["week", "model"]):
        valid = cell[cell.protocol.eq("cutoff_valid")]
        if valid.empty:
            continue
        capacity = _capacity(valid, budget)
        # Every adverse outcome in the eligible population is the denominator
        # for coverage-adjusted recall, reachable or not.
        adverse_valid = int(valid.risk.sum())
        n_valid = len(valid)

        arms = {
            "A->A  activity-conditioned":
                cell[cell.protocol.eq("activity_conditioned")],
            "A->A  active-and-eligible":
                cell[cell.protocol.eq("activity_conditioned") & cell.cutoff_valid],
            # Same activity-trained estimator, valid candidate set: this is the
            # arm that isolates the candidate set rather than the estimator.
            "A->V  activity-trained, valid set":
                cell[cell.protocol.eq("activity_to_cutoff_valid")],
            "V->V  cutoff-valid (reference)":
                valid,
        }
        for name, frame in arms.items():
            if frame.empty:
                continue
            chosen = _select(frame, capacity)
            n_sel = len(chosen)
            eligible_sel = chosen[chosen.cutoff_valid]
            reached = int(eligible_sel.risk.sum())
            rows.append({
                "week": week, "model": model, "arm": name,
                "candidate_coverage": len(frame[frame.cutoff_valid]) / n_valid,
                "noneligible_allocation": 1 - len(eligible_sel) / n_sel if n_sel else np.nan,
                "eligibility_adjusted_precision": reached / n_sel if n_sel else np.nan,
                "coverage_adjusted_recall": reached / adverse_valid if adverse_valid else np.nan,
                # Adverse silent learners the arm could never score.
                "missed_adverse_silent": int(
                    valid[~valid[KEY].apply(tuple, axis=1).isin(
                        set(map(tuple, frame[KEY].to_numpy()))
                    ) & valid.risk.eq(1)].shape[0]
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("results/cohort_exchange_2x2_clean/oulab"))
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    table = evaluate(args.results, args.budget)
    summary = table.groupby("arm").agg(
        candidate_coverage=("candidate_coverage", "median"),
        noneligible_allocation=("noneligible_allocation", "median"),
        elig_adj_precision=("eligibility_adjusted_precision", "median"),
        cov_adj_recall=("coverage_adjusted_recall", "median"),
        missed_adverse_silent=("missed_adverse_silent", "median"),
    )
    print(f"\nBudget {args.budget:.0%}, medians across {table.week.nunique()} "
          f"landmarks x {table.model.nunique()} models\n")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nBy landmark (coverage-adjusted recall):")
    piv = table.pivot_table(index="week", columns="arm",
                            values="coverage_adjusted_recall", aggfunc="median")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    # Paired within-cell decomposition. Differences of separate medians are not
    # additive, so the attribution is computed per model-landmark cell first.
    wide = table.pivot_table(index=["week", "model"], columns="arm",
                             values="coverage_adjusted_recall")
    aa = "A->A  activity-conditioned"
    inter = "A->A  active-and-eligible"
    av = "A->V  activity-trained, valid set"
    if {aa, inter, av}.issubset(wide.columns):
        gains = pd.DataFrame({
            "eligibility": wide[inter] - wide[aa],
            "silence": wide[av] - wide[inter],
        })
        gains["total"] = gains.eligibility + gains.silence
        print(f"\nPaired within-cell gains in coverage-adjusted recall "
              f"(n={len(gains)} cells):")
        for name in ["eligibility", "silence", "total"]:
            col = gains[name]
            print(f"  {name:<12} median={col.median():+.4f}  mean={col.mean():+.4f}"
                  f"  IQR=[{col.quantile(.25):+.4f},{col.quantile(.75):+.4f}]"
                  f"  positive in {100 * (col > 0).mean():.0f}% of cells")
        share = gains.eligibility.sum() / gains.total.sum()
        print(f"  eligibility share of total recovered recall: {100 * share:.1f}%")


if __name__ == "__main__":
    main()
