"""Paired aggregate inference for the main and learner-disjoint experiments.

Uses saved predictions only. One canonical vector of named-presentation
multiplicities is shared by every model, landmark, and fold-grouping arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_population_components import prediction_cells, AA, AV, VA, VV
from src.cohort_exchange import METRICS, metric_values, _bootstrap_cluster_counts, _weighted_cluster_metric_draws

SPECS = {
    "joint_population_difference": {AA: 1., VV: -1.},
    "evaluation_population_shapley": {AA: .5, AV: -.5, VA: .5, VV: -.5},
    "training_population_shapley": {AA: .5, VA: -.5, AV: .5, VV: -.5},
}


def cell_estimates(frame, counts, order):
    required = {AA, AV, VA, VV}
    if not required.issubset(frame.protocol):
        raise ValueError("A prediction cell lacks one or more required 2x2 protocols")
    relevant = frame[frame.protocol.isin(required)]
    if relevant.cluster_id.isna().any() or set(relevant.cluster_id.astype(str)) != set(order):
        raise ValueError("Every OULAD cell must contain the same named presentation set")
    protocol_points, protocol_draws = {}, {}
    for protocol in required:
        rows = relevant[relevant.protocol.eq(protocol)]
        protocol_points[protocol] = metric_values(rows)
        protocol_draws[protocol] = _weighted_cluster_metric_draws(rows, counts, order)
    return {
        (metric, contrast): {
            "point": sum(coef * protocol_points[protocol][metric] for protocol, coef in coefficients.items()),
            "draws": sum(coef * protocol_draws[protocol][metric] for protocol, coef in coefficients.items()),
        }
        for metric in METRICS for contrast, coefficients in SPECS.items()
    }


def summarize(cells, arm):
    if not cells:
        raise ValueError(f"No cells available for {arm}")
    rows, vectors = [], {}
    for metric in METRICS:
        for contrast in SPECS:
            key = (metric, contrast)
            values = np.mean([cell[key]["draws"] for cell in cells.values()], axis=0)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"No finite aggregate draws: {arm}/{key}")
            point = float(np.mean([cell[key]["point"] for cell in cells.values()]))
            lo, hi = np.quantile(finite, [.025, .975])
            rows.append(dict(arm=arm, cells=len(cells), metric=metric, contrast=contrast,
                             point_mean=point, bootstrap_mean=float(finite.mean()),
                             ci_low=float(lo), ci_high=float(hi), finite_draws=len(finite)))
            vectors[key] = (point, values)
        eval_point, eval_draws = vectors[(metric, "evaluation_population_shapley")]
        train_point, train_draws = vectors[(metric, "training_population_shapley")]
        contrast = "absolute_evaluation_minus_absolute_training"
        values = np.abs(eval_draws)-np.abs(train_draws)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"No finite magnitude-difference draws: {arm}/{metric}")
        point = abs(eval_point)-abs(train_point)
        lo, hi = np.quantile(finite, [.025, .975])
        rows.append(dict(arm=arm, cells=len(cells), metric=metric, contrast=contrast,
                         point_mean=point, bootstrap_mean=float(finite.mean()),
                         ci_low=float(lo), ci_high=float(hi), finite_draws=len(finite)))
        vectors[(metric, contrast)] = (point, values)
    return rows, vectors


def expected_keys(folder):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    return {(int(week), model) for week in protocol["weeks"] for model in protocol["models"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.n_boot < 1 or args.threads < 1:
        raise ValueError("n-boot and threads must be positive")
    estimates, order, counts = {}, None, None
    with threadpool_limits(limits=args.threads):
        for arm, folder in (("main", args.main), ("learner_disjoint", args.learner)):
            expected, cells = expected_keys(folder), {}
            for frame in prediction_cells(folder):
                key = (int(frame.week.iloc[0]), str(frame.model.iloc[0]))
                if key in cells or key not in expected:
                    raise ValueError(f"Unexpected or duplicate {arm} cell {key}")
                if order is None:
                    relevant = frame[frame.protocol.isin((AA, AV, VA, VV))]
                    order = np.array(sorted(relevant.cluster_id.astype(str).unique()))
                    counts = _bootstrap_cluster_counts(args.n_boot, len(order), np.random.default_rng(args.seed))
                cells[key] = cell_estimates(frame, counts, order)
                print(f"{arm}: {len(cells)}/{len(expected)} cells", flush=True)
            if set(cells) != expected:
                raise ValueError(f"Missing {arm} cells: {sorted(expected.difference(cells))}")
            estimates[arm] = cells
    learner_keys = set(estimates["learner_disjoint"])
    if not learner_keys.issubset(estimates["main"]):
        raise ValueError("Learner-disjoint cells must be present in the main experiment")
    estimates["main_matched4"] = {key: estimates["main"][key] for key in sorted(learner_keys)}
    outputs, vectors = {}, {}
    for arm, cells in estimates.items():
        outputs[arm], vectors[arm] = summarize(cells, "main" if arm == "main_matched4" else arm)
    difference = []
    for metric in METRICS:
        for contrast in SPECS:
            key = (metric, contrast)
            learner_point, learner_draws = vectors["learner_disjoint"][key]
            main_point, main_draws = vectors["main_matched4"][key]
            draws = learner_draws-main_draws
            draws = draws[np.isfinite(draws)]
            if not len(draws):
                raise ValueError(f"No finite paired difference draws: {key}")
            lo, hi = np.quantile(draws, [.025, .975])
            difference.append(dict(metric=metric, contrast=contrast,
                point_learner_minus_matched_main=learner_point-main_point,
                ci_low=float(lo), ci_high=float(hi)))
    outputs["learner_minus_matched_main"] = difference
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        (args.output / f"aggregate_{name}_final.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    (args.output / "aggregate_comparison_protocol.json").write_text(json.dumps({
        "n_boot": args.n_boot, "seed": args.seed,
        "canonical_cluster_order": order.tolist(), "model_refits": 0,
        "risk_score": "1 - p_success", "point_estimate": "observed mean across fixed cells",
    }, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
