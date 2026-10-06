"""Recompute eligibility-first contrasts from saved predictions without fitting.

Requires local predictions or completed per-cell checkpoints. Writes only
aggregate metric components, candidate-set budgets, and paired recall gains.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cohort_exchange import metric_values, select_at_budget, _capacity_by_cluster

AA, AV, VA, VV = ("activity_conditioned", "activity_to_cutoff_valid",
                  "cutoff_valid_to_activity", "cutoff_valid")
COLUMNS = ["week", "model", "protocol", "y", "p_success", "risk", "risk_score",
           "cutoff_valid", "cluster_id", "id_student", "code_module", "code_presentation"]


def prediction_cells(results: Path):
    checkpoints = sorted((results / "checkpoints").glob("week_*__*.csv.gz"))
    if checkpoints:
        for path in checkpoints:
            frame = pd.read_csv(path, usecols=COLUMNS, float_precision="round_trip")
            yield from (cell for _, cell in frame.groupby(["week", "model"], sort=True))
    else:
        path = results / "predictions.csv.gz"
        if not path.is_file():
            raise FileNotFoundError("This analysis needs predictions.csv.gz or completed per-cell checkpoints from a local run")
        frame = pd.read_csv(path, usecols=COLUMNS, float_precision="round_trip")
        yield from (cell for _, cell in frame.groupby(["week", "model"], sort=True))


def cell_components(frame: pd.DataFrame, budget: float):
    missing = {AA, AV, VA, VV}.difference(frame.protocol)
    if missing:
        raise ValueError(f"Incomplete training-by-evaluation cell: missing {sorted(missing)}")
    if not 0 < budget <= 1:
        raise ValueError("budget must lie in (0, 1]")
    week, model = int(frame.week.iloc[0]), str(frame.model.iloc[0])
    arms = {protocol: frame[frame.protocol.eq(protocol)] for protocol in (AA, AV, VA, VV)}
    arms["AI"] = arms[AA][arms[AA].cutoff_valid]
    arms["VI"] = arms[VA][arms[VA].cutoff_valid]
    scores = {arm: metric_values(rows) for arm, rows in arms.items()}
    metric_rows = []
    for metric in ("auc", "pr_auc_risk", "brier", "ece"):
        s = {arm: values[metric] for arm, values in scores.items()}
        metric_rows.append(dict(
            week=week, model=model, metric=metric,
            nonelig=.5*((s[AA]-s["AI"])+(s[VA]-s["VI"])),
            silence=.5*((s["AI"]-s[AV])+(s["VI"]-s[VV])),
            evaluation=.5*((s[AA]-s[AV])+(s[VA]-s[VV])),
        ))
    reference_risks = float(arms[VV].risk.sum())
    if not reference_risks:
        raise ValueError(f"Week {week}/{model}: no adverse outcomes in the valid reference population")
    capacities = _capacity_by_cluster(arms[VV], budget)
    recalls, budget_rows = {}, []
    for arm in (AA, "AI", AV, VV):
        selected = select_at_budget(arms[arm], capacities)
        if selected.empty:
            raise ValueError(f"Week {week}/{model}: candidate arm {arm} selects no learners")
        eligible = selected[selected.cutoff_valid]
        reached = float(eligible.risk.sum())
        recalls[arm] = reached/reference_risks
        budget_rows.append(dict(
            week=week, model=model, arm=arm, budget=budget,
            coverage_adjusted_recall=recalls[arm],
            actionable_precision=reached/len(selected),
            wasted_budget_rate=1-len(eligible)/len(selected),
            candidate_coverage=float(arms[arm].cutoff_valid.sum())/len(arms[VV]),
        ))
    gain = dict(week=week, model=model, budget=budget,
                eligibility=recalls["AI"]-recalls[AA],
                silence=recalls[AV]-recalls["AI"], total=recalls[AV]-recalls[AA])
    return metric_rows, budget_rows, gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Aggregate output directory (default: results directory)")
    parser.add_argument("--budget", type=float, default=.05)
    args = parser.parse_args()
    metric_rows, budget_rows, gains, seen = [], [], [], set()
    for cell in prediction_cells(args.results):
        key = (int(cell.week.iloc[0]), str(cell.model.iloc[0]))
        if key in seen:
            raise ValueError(f"Duplicate prediction cell: {key}")
        seen.add(key)
        metric, budget, gain = cell_components(cell, args.budget)
        metric_rows.extend(metric)
        budget_rows.extend(budget)
        gains.append(gain)
    if not seen:
        raise ValueError("No prediction cells were found")
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("secondary_metric_components", metric_rows),
                       ("active_eligible_budget", budget_rows), ("paired_recall_gains", gains)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    gain_table = pd.DataFrame(gains)
    print(f"Computed {len(seen)} configuration-landmark cells at budget {args.budget:g}")
    for name in ("eligibility", "silence", "total"):
        values = gain_table[name]
        print(f"{name}: median={values.median():+.6f}; positive={values.gt(0).sum()}, "
              f"negative={values.lt(0).sum()}, zero={values.eq(0).sum()}")


if __name__ == "__main__":
    main()
