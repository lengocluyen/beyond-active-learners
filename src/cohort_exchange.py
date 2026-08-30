"""Cohort-validity benchmark for longitudinal early-warning systems.

The benchmark separates four questions that are often conflated:

``activity_conditioned``
    Who remains after feature construction starts from an event table?
``static_full``
    What happens when every labelled enrolment is retained, even when the
    outcome may already have occurred?  This is a sensitivity analysis.
``cutoff_valid``
    Who was registered and still eligible for intervention at the landmark?
    Digitally silent learners remain in this risk set.
``discrete_hazard``
    Among learners in the cutoff-valid risk set, who withdraws in the next
    fixed interval?  This is a pooled person-week discrete-time hazard arm.

OULAD exposes registration and withdrawal dates and supports all four arms.
KDD Cup 2015 has final dropout labels but no withdrawal dates, so only
``activity_conditioned`` and ``static_full`` are identified.

All classifiers use presentation-grouped folds.  Fold assignments are created
once from the labelled roster and reused across protocols.  For uncertainty,
repeated out-of-fold predictions are first averaged per learner; presentation
clusters are then bootstrapped.  CV folds are never treated as independent
sampling units.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .full_evaluation import (
    REFERENCE_BASELINE_METADATA,
    _expected_calibration_error,
    _model,
    _representations,
)
from .paths import get_data_path
from .pcg_ut import KEY, _load_weekly_events


PRESENTATION = ["code_module", "code_presentation"]
PROTOCOLS = ("activity_conditioned", "static_full", "cutoff_valid")
CROSS_PROTOCOL = "activity_to_cutoff_valid"
CROSS_PROTOCOL_VA = "cutoff_valid_to_activity"
# Named train/evaluation cells that are not simply "train and score on the same
# population".  Mapping cell -> (training population, evaluation population).
CROSS_CELL_POPULATIONS = {
    CROSS_PROTOCOL: ("activity_conditioned", "cutoff_valid"),
    CROSS_PROTOCOL_VA: ("cutoff_valid", "activity_conditioned"),
}
DEFAULT_WEEKS = {
    "oulab": [2, 4, 6, 8, 10, 12, 14, 16],
    "kdd": [1, 2, 3, 4, 5],
}
DEFAULT_MODELS = {
    # All three resolve to the matched HGB head in full_evaluation._model.
    "oulab": ("flat_hgb", "temporal_hgb", "stack_7_full"),
    "kdd": ("flat_hgb", "temporal_hgb", "pcg_ut"),
}
DEFAULT_BUDGETS = (0.01, 0.05, 0.10, 0.20)
METRICS = ("auc", "pr_auc_risk", "brier", "ece")


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    weeks: tuple[int, ...]
    models: tuple[str, ...]
    folds: int = 5
    repeats: int = 5
    seed: int = 42
    budgets: tuple[float, ...] = DEFAULT_BUDGETS
    bootstrap_iterations: int = 1000
    hazard_days: int = 7
    cluster: str = "presentation"
    split_unit: str = "presentation"
    jobs: int = 1


def _set_dataset(dataset: str) -> None:
    if dataset not in DEFAULT_WEEKS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {tuple(DEFAULT_WEEKS)}")
    os.environ["PCG_DATASET"] = dataset


def _group_id(frame: pd.DataFrame, cluster: str = "presentation") -> pd.Series:
    if cluster == "module":
        return frame["code_module"].astype(str)
    if cluster == "learner":
        # Learner-disjoint splitting. 12.3% of OULAD learners appear in more
        # than one presentation, so presentation groups do not guarantee that a
        # learner is absent from training when their record is held out.
        return frame["id_student"].astype(str)
    if cluster != "presentation":
        raise ValueError(
            "cluster must be 'presentation', 'module' or 'learner'"
        )
    return (
        frame["code_module"].astype(str)
        + "::"
        + frame["code_presentation"].astype(str)
    )


def load_roster(dataset: str) -> pd.DataFrame:
    """Return one labelled row per learner-presentation with event-time metadata."""
    _set_dataset(dataset)
    labels = pd.read_csv(get_data_path("processed/labels.csv"))
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    roster = labels[KEY + ["label"]].drop_duplicates(KEY).copy()

    if dataset == "oulab":
        registration = pd.read_csv(
            get_data_path("raw/studentRegistration.csv"),
            usecols=KEY + ["date_registration", "date_unregistration"],
        )
        registration["date_registration"] = pd.to_numeric(
            registration["date_registration"], errors="coerce"
        )
        registration["date_unregistration"] = pd.to_numeric(
            registration["date_unregistration"], errors="coerce"
        )
        roster = roster.merge(registration, on=KEY, how="left", validate="one_to_one")
    else:
        roster["date_registration"] = np.nan
        roster["date_unregistration"] = np.nan

    roster["risk"] = 1 - roster["label"]
    return roster


def cohort_membership(
    dataset: str,
    week: int,
    roster: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify every labelled enrolment relative to a prediction landmark."""
    _set_dataset(dataset)
    roster = load_roster(dataset) if roster is None else roster.copy()
    events = _load_weekly_events(week) if events is None else events
    active = events[KEY].drop_duplicates().assign(has_activity=True)
    frame = roster.merge(active, on=KEY, how="left", validate="one_to_one")
    # The right-hand marker is True for an observed event and missing after the
    # left join otherwise.  ``notna`` expresses that directly and avoids
    # pandas' deprecated silent object downcast in ``fillna(False)``.
    frame["has_activity"] = frame["has_activity"].notna()
    frame["no_activity"] = (~frame["has_activity"]).astype(int)
    cutoff_day = 7 * int(week)
    frame["cutoff_day"] = cutoff_day

    if dataset == "oulab":
        frame["registered_by_cutoff"] = (
            frame["date_registration"].isna()
            | (frame["date_registration"] < cutoff_day)
        )
        frame["outcome_realized"] = (
            frame["date_unregistration"].notna()
            & (frame["date_unregistration"] < cutoff_day)
        )
        frame["cutoff_valid"] = (
            frame["registered_by_cutoff"] & ~frame["outcome_realized"]
        )
        frame["membership_class"] = np.select(
            [
                ~frame["registered_by_cutoff"],
                frame["outcome_realized"] & frame["has_activity"],
                frame["outcome_realized"] & ~frame["has_activity"],
                frame["cutoff_valid"] & frame["has_activity"],
                frame["cutoff_valid"] & ~frame["has_activity"],
            ],
            [
                "not_yet_registered",
                "outcome_realized_active",
                "outcome_realized_silent",
                "eligible_active",
                "eligible_silent",
            ],
            default="unclassified",
        )
    else:
        # KDD contains final dropout labels but no dropout event time.  The full
        # enrolment roster is therefore the only identifiable risk-set proxy.
        frame["registered_by_cutoff"] = True
        frame["outcome_realized"] = False
        frame["cutoff_valid"] = True
        frame["membership_class"] = np.where(
            frame["has_activity"], "eligible_active", "eligible_silent"
        )

    frame["activity_conditioned"] = frame["has_activity"]
    frame["static_full"] = True
    frame["dataset"] = dataset
    frame["week"] = int(week)
    return frame


def composition_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Detailed cohort flow plus validity indices for one or more landmarks."""
    detailed = (
        membership.groupby(["dataset", "week", "membership_class"], dropna=False)
        .agg(n=("label", "size"), risk_rate=("risk", "mean"))
        .reset_index()
    )
    totals = (
        membership.groupby(["dataset", "week"])
        .agg(
            n_labelled=("label", "size"),
            n_activity=("activity_conditioned", "sum"),
            n_static=("static_full", "sum"),
            n_cutoff_valid=("cutoff_valid", "sum"),
            n_outcome_realized=("outcome_realized", "sum"),
        )
        .reset_index()
    )

    rows = []
    for (dataset, week), group in membership.groupby(["dataset", "week"]):
        active = group["activity_conditioned"]
        valid = group["cutoff_valid"]
        realized_in_active = active & group["outcome_realized"]
        silent_valid = valid & ~group["has_activity"]
        active_valid = valid & group["has_activity"]
        active_keys = set(map(tuple, group.loc[active, KEY].to_numpy()))
        valid_keys = set(map(tuple, group.loc[valid, KEY].to_numpy()))
        union = len(active_keys | valid_keys)
        rows.append(
            {
                "dataset": dataset,
                "week": week,
                "n_activity": int(active.sum()),
                "n_cutoff_valid": int(valid.sum()),
                "known_outcomes_in_activity": int(realized_in_active.sum()),
                "eligible_silent_excluded": int(silent_valid.sum()),
                "eligible_active": int(active_valid.sum()),
                "activity_contamination_rate": (
                    float(realized_in_active.sum() / active.sum()) if active.any() else np.nan
                ),
                "silent_exclusion_rate": (
                    float(silent_valid.sum() / valid.sum()) if valid.any() else np.nan
                ),
                "cohort_jaccard": (
                    float(len(active_keys & valid_keys) / union) if union else np.nan
                ),
                "silent_risk_rate": (
                    float(group.loc[silent_valid, "risk"].mean())
                    if silent_valid.any()
                    else np.nan
                ),
                "eligible_active_risk_rate": (
                    float(group.loc[active_valid, "risk"].mean())
                    if active_valid.any()
                    else np.nan
                ),
                "risk_concentration_ratio": (
                    float(
                        group.loc[silent_valid, "risk"].mean()
                        / group.loc[active_valid, "risk"].mean()
                    )
                    if silent_valid.any()
                    and active_valid.any()
                    and group.loc[active_valid, "risk"].mean() > 0
                    else np.nan
                ),
            }
        )
    indices = pd.DataFrame(rows)
    detailed.attrs["totals"] = totals
    detailed.attrs["indices"] = indices
    return detailed


def _supplement_sources(dataset: str, week: int) -> list[pd.DataFrame]:
    """Feature blocks available even when a learner has no event-derived row."""
    if dataset != "oulab":
        return []
    sources: list[pd.DataFrame] = []
    for relative in (
        "processed/static_features.csv",
        f"processed/behavioral_week{week}.csv",
        f"processed/traversal_week{week}.csv",
    ):
        path = get_data_path(relative)
        if path.exists():
            sources.append(pd.read_csv(path))
    return sources


def align_features_to_roster(
    dataset: str,
    week: int,
    roster: pd.DataFrame,
    active_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Left-align a representation and recover roster-grounded feature blocks.

    Event-derived graph/temporal values become zero for silent learners.
    Enrolment and assessment-schedule values are recovered from their native
    roster-based blocks rather than being incorrectly erased.
    """
    feature_cols = [
        column for column in active_features.columns
        if column not in KEY + ["snapshot_week"]
    ]
    aligned = roster[KEY].merge(active_features, on=KEY, how="left")

    for source in _supplement_sources(dataset, week):
        usable = [column for column in feature_cols if column in source.columns]
        if not usable:
            continue
        supplement = source[KEY + usable].drop_duplicates(KEY)
        aligned = aligned.merge(supplement, on=KEY, how="left", suffixes=("", "__supp"))
        for column in usable:
            supplement_column = f"{column}__supp"
            if supplement_column in aligned:
                aligned[column] = aligned[column].combine_first(aligned[supplement_column])
                aligned = aligned.drop(columns=supplement_column)

    aligned["no_activity"] = (~aligned[KEY].apply(tuple, axis=1).isin(
        set(map(tuple, active_features[KEY].to_numpy()))
    )).astype(int)
    if "no_activity" not in feature_cols:
        feature_cols.append("no_activity")
    aligned[feature_cols] = (
        aligned[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return aligned[KEY + feature_cols], feature_cols


def make_fold_maps(
    roster: pd.DataFrame,
    n_splits: int,
    repeats: int,
    seed: int,
    split_unit: str = "presentation",
) -> dict[int, dict[str, int]]:
    """Assign every split group to a fold once per repeat."""
    groups = _group_id(roster, split_unit)
    if groups.nunique() < n_splits:
        raise ValueError(
            f"only {groups.nunique()} {split_unit} groups; cannot use {n_splits} folds"
        )
    y = roster["label"].astype(int).to_numpy()
    design = np.zeros((len(roster), 1))
    maps: dict[int, dict[str, int]] = {}
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        mapping: dict[str, int] = {}
        for fold, (_, test_index) in enumerate(
            splitter.split(design, y, groups), start=1
        ):
            for group in groups.iloc[test_index].unique():
                mapping[str(group)] = fold
        maps[repeat + 1] = mapping
    return maps


def _protocols_for_dataset(dataset: str) -> tuple[str, ...]:
    return PROTOCOLS if dataset == "oulab" else PROTOCOLS[:2]


def _prediction_protocols_for_dataset(dataset: str) -> tuple[str, ...]:
    """Named train/evaluation cells emitted by the landmark benchmark."""
    protocols = _protocols_for_dataset(dataset)
    if dataset == "oulab":
        return (*protocols, CROSS_PROTOCOL, CROSS_PROTOCOL_VA)
    return protocols


def _train_evaluation_specs(
    dataset: str,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map each fitted training population to its evaluation populations.

    Values contain ``(output_protocol, evaluation_population)`` pairs.  Each
    estimator is fitted once per fold and scored on both populations, giving
    the full 2x2 training-by-evaluation design: the activity-conditioned fit
    produces A->A and A->V, and the cutoff-valid fit produces V->V and V->A.
    Holding a fitted estimator fixed across two evaluation populations makes
    each within-row contrast a pure evaluation-population contrast, and each
    within-column contrast a pure training-population contrast.
    """
    specs: dict[str, tuple[tuple[str, str], ...]] = {
        "activity_conditioned": (("activity_conditioned", "activity_conditioned"),),
        "static_full": (("static_full", "static_full"),),
    }
    if dataset == "oulab":
        specs["activity_conditioned"] = (
            ("activity_conditioned", "activity_conditioned"),
            (CROSS_PROTOCOL, "cutoff_valid"),
        )
        specs["cutoff_valid"] = (
            ("cutoff_valid", "cutoff_valid"),
            (CROSS_PROTOCOL_VA, "activity_conditioned"),
        )
    return specs


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV checkpoint atomically so interruptions cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if path.suffix == ".gz" else ".tmp"
    temporary = path.with_name(path.name + suffix)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _landmark_checkpoint_path(
    checkpoint_dir: Path,
    week: int,
    model_name: str,
) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return checkpoint_dir / f"week_{int(week):03d}__{safe_model}.csv.gz"


def _limit_estimator_inner_jobs(estimator: object) -> object:
    """Avoid nested thread/process pools during fold-level parallelism."""
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return estimator
    parameters = estimator.get_params(deep=True)
    updates = {
        name: 1 for name in parameters
        if name == "n_jobs" or name.endswith("__n_jobs")
    }
    if updates:
        estimator.set_params(**updates)
    return estimator


def run_landmark_benchmark(
    config: BenchmarkConfig,
    verbose: bool = False,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    fit_specs: dict[str, tuple[tuple[str, str], ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit matched landmark models and return averaged predictions.

    Each completed week/model cell is averaged across repeats and atomically
    checkpointed when ``checkpoint_dir`` is supplied.  A resumed run therefore
    skips completed fits and never needs to retain all repeat-level predictions
    in memory.
    """
    _set_dataset(config.dataset)
    roster = load_roster(config.dataset)
    fold_maps = make_fold_maps(
        roster, config.folds, config.repeats, config.seed, config.split_unit
    )
    prediction_cells: list[pd.DataFrame] = []
    membership_rows: list[pd.DataFrame] = []
    fit_specs = fit_specs or _train_evaluation_specs(config.dataset)
    n_output_cells = sum(len(evaluations) for evaluations in fit_specs.values())

    for week in config.weeks:
        events = _load_weekly_events(int(week))
        membership = cohort_membership(config.dataset, int(week), roster, events)
        membership_rows.append(membership)
        if verbose:
            print(
                f"[cohort] {config.dataset} week={week}: "
                f"{len(config.models)} models x "
                f"{n_output_cells} "
                "train/evaluation cells",
                flush=True,
            )

        for model_index, model_name in enumerate(config.models, start=1):
            checkpoint = (
                _landmark_checkpoint_path(checkpoint_dir, int(week), model_name)
                if checkpoint_dir is not None else None
            )
            if resume and checkpoint is not None and checkpoint.exists():
                cached = pd.read_csv(checkpoint)
                prediction_cells.append(cached)
                if verbose:
                    print(
                        f"[resume] {config.dataset} week={week} "
                        f"model={model_name} rows={len(cached):,}",
                        flush=True,
                    )
                continue

            if verbose:
                print(
                    f"[model] {config.dataset} week={week} "
                    f"{model_index}/{len(config.models)} model={model_name}",
                    flush=True,
                )
            # Construct one representation at a time.  This bounds peak memory
            # for KDD and makes the checkpoint boundary meaningful.
            representations = _representations(
                events, int(week), seed=config.seed, models=(model_name,)
            )
            active_features = representations[model_name]
            aligned, feature_cols = align_features_to_roster(
                config.dataset, int(week), roster, active_features
            )
            # Membership owns the canonical eligibility-derived flag. The
            # alignment helper also returns it for standalone callers.
            base = membership.merge(
                aligned.drop(columns=["no_activity"], errors="ignore"),
                on=KEY,
                how="left",
                validate="one_to_one",
            )
            # Split groups default to presentations.  The inferential cluster
            # may be coarser (module) but must never alter train/test isolation.
            # ``split_unit="learner"`` is the learner-disjoint sensitivity arm.
            base["_split_group"] = _group_id(base, config.split_unit)
            base["_cluster"] = _group_id(base, config.cluster)
            model_rows: list[pd.DataFrame] = []

            for repeat, mapping in fold_maps.items():
                base["_fold"] = base["_split_group"].map(mapping)
                for train_protocol, evaluation_specs in fit_specs.items():
                    if verbose:
                        evaluated = ",".join(
                            protocol for protocol, _ in evaluation_specs
                        )
                        print(
                            f"[fit] week={week} model={model_name} "
                            f"train={train_protocol} evaluate={evaluated} "
                            f"repeat={repeat}/{config.repeats}",
                            flush=True,
                        )

                    def fit_fold(fold: int) -> list[pd.DataFrame]:
                        train = (
                            base[train_protocol].astype(bool)
                            & base["_fold"].ne(fold)
                        )
                        y_train = base.loc[train, "label"].astype(int)
                        if len(y_train) == 0 or y_train.nunique() < 2:
                            return []
                        estimator = _model(
                            model_name, config.seed + repeat * 100 + fold
                        )
                        if config.jobs != 1:
                            estimator = _limit_estimator_inner_jobs(estimator)
                        estimator.fit(base.loc[train, feature_cols], y_train)
                        records: list[pd.DataFrame] = []
                        for protocol, evaluation_protocol in evaluation_specs:
                            test = (
                                base[evaluation_protocol].astype(bool)
                                & base["_fold"].eq(fold)
                            )
                            y_test = base.loc[test, "label"].astype(int)
                            if len(y_test) == 0 or y_test.nunique() < 2:
                                continue
                            probability = estimator.predict_proba(
                                base.loc[test, feature_cols]
                            )[:, 1]
                            record = base.loc[
                                test,
                                KEY
                                + [
                                    "label",
                                    "risk",
                                    "has_activity",
                                    "no_activity",
                                    "outcome_realized",
                                    "cutoff_valid",
                                    "membership_class",
                                    "_cluster",
                                ],
                            ].copy()
                            record = record.rename(
                                columns={"label": "y", "_cluster": "cluster_id"}
                            )
                            record["p_success"] = probability
                            record["risk_score"] = 1.0 - probability
                            record["dataset"] = config.dataset
                            record["week"] = int(week)
                            record["protocol"] = protocol
                            record["train_protocol"] = train_protocol
                            record["eval_protocol"] = evaluation_protocol
                            record["model"] = model_name
                            record["repeat"] = repeat
                            record["fold"] = fold
                            record["cluster_unit"] = config.cluster
                            records.append(record)
                        return records

                    inner_threads = max(
                        1, int(os.environ.get("PCG_BLAS_THREADS", "1"))
                    )
                    # threadpoolctl constrains already-loaded BLAS/OpenMP
                    # libraries, complementing the early environment limits in
                    # the CLI. This prevents native thread-pool multiplication
                    # when folds run concurrently.
                    with threadpool_limits(limits=inner_threads):
                        fold_record_groups = Parallel(
                            n_jobs=config.jobs,
                            prefer="threads",
                        )(
                            delayed(fit_fold)(fold)
                            for fold in range(1, config.folds + 1)
                        )
                    model_rows.extend(
                        record
                        for records in fold_record_groups
                        for record in records
                    )

            if model_rows:
                averaged_cell = average_repeated_predictions(
                    pd.concat(model_rows, ignore_index=True)
                )
            else:
                averaged_cell = pd.DataFrame()
            prediction_cells.append(averaged_cell)
            if checkpoint is not None and not averaged_cell.empty:
                _atomic_to_csv(averaged_cell, checkpoint)
            if verbose:
                print(
                    f"[checkpoint] {config.dataset} week={week} "
                    f"model={model_name} rows={len(averaged_cell):,}",
                    flush=True,
                )

    predictions = (
        pd.concat(prediction_cells, ignore_index=True)
        if prediction_cells else pd.DataFrame()
    )
    memberships = pd.concat(membership_rows, ignore_index=True)
    detailed = composition_table(memberships)
    indices = detailed.attrs["indices"]
    return predictions, memberships, pd.concat(
        [
            detailed.assign(table="flow"),
            indices.assign(membership_class=np.nan, table="indices"),
        ],
        ignore_index=True,
        sort=False,
    )


def _with_train_eval_protocols(predictions: pd.DataFrame) -> pd.DataFrame:
    """Backfill explicit train/evaluation labels on legacy prediction files."""
    predictions = predictions.copy()
    protocol = predictions["protocol"]
    if "train_protocol" not in predictions:
        predictions["train_protocol"] = protocol.map(
            lambda cell: CROSS_CELL_POPULATIONS.get(cell, (cell, cell))[0]
        )
    if "eval_protocol" not in predictions:
        predictions["eval_protocol"] = protocol.map(
            lambda cell: CROSS_CELL_POPULATIONS.get(cell, (cell, cell))[1]
        )
    return predictions


def average_repeated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average stochastic repeats before any inferential resampling."""
    if predictions.empty:
        return predictions.copy()
    predictions = _with_train_eval_protocols(predictions)
    group_cols = [
        "dataset", "week", "protocol", "train_protocol", "eval_protocol",
        "model", *KEY,
        "y", "risk", "has_activity", "no_activity", "outcome_realized",
        "cutoff_valid", "membership_class", "cluster_id", "cluster_unit",
    ]
    return (
        predictions.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            p_success=("p_success", "mean"),
            risk_score=("risk_score", "mean"),
            prediction_sd=("p_success", "std"),
            n_repeats=("repeat", "nunique"),
        )
    )


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y"].astype(int).to_numpy()
    p = frame["p_success"].astype(float).to_numpy()
    if len(frame) == 0 or np.unique(y).size < 2:
        return {metric: np.nan for metric in METRICS}
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc_risk": float(average_precision_score(1 - y, 1 - p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(_expected_calibration_error(y, p)),
    }


def _cluster_metric_components(
    frame: pd.DataFrame,
    cluster_order: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pre-aggregate one prediction cell for fast weighted cluster draws.

    Resampling a cluster multiple times is exactly equivalent to assigning all
    of its learner rows that integer frequency as a sample weight.  Aggregating
    by score threshold, calibration bin, and cluster lets every bootstrap draw
    use small matrix operations instead of copying and sorting the full table.
    """
    cluster_lookup = {str(value): index for index, value in enumerate(cluster_order)}
    cluster_codes = np.fromiter(
        (cluster_lookup[str(value)] for value in frame["cluster_id"]),
        dtype=np.int64,
        count=len(frame),
    )
    y = frame["y"].to_numpy(dtype=np.int8)
    p = frame["p_success"].to_numpy(dtype=float)
    n_clusters = len(cluster_order)
    score_order = np.argsort(p, kind="mergesort")
    sorted_p = p[score_order]
    score_starts = np.r_[
        0, 1 + np.flatnonzero(sorted_p[1:] != sorted_p[:-1])
    ]

    squared_error = np.bincount(
        cluster_codes, weights=(y - p) ** 2, minlength=n_clusters
    )
    cluster_n = np.bincount(cluster_codes, minlength=n_clusters).astype(float)

    # Match full_evaluation._expected_calibration_error: ten equal-width bins,
    # with p=1 assigned to the final bin.
    calibration_codes = np.digitize(p, np.linspace(0.0, 1.0, 11)[1:-1])
    flat_bin_cluster = calibration_codes * n_clusters + cluster_codes
    bin_shape = (10, n_clusters)
    bin_n = np.bincount(
        flat_bin_cluster, minlength=10 * n_clusters
    ).reshape(bin_shape).astype(float)
    bin_y = np.bincount(
        flat_bin_cluster, weights=y, minlength=10 * n_clusters
    ).reshape(bin_shape)
    bin_p = np.bincount(
        flat_bin_cluster, weights=p, minlength=10 * n_clusters
    ).reshape(bin_shape)
    return {
        "sorted_y": y[score_order].astype(float),
        "sorted_cluster": cluster_codes[score_order],
        "score_starts": score_starts,
        "squared_error": squared_error,
        "cluster_n": cluster_n,
        "bin_n": bin_n,
        "bin_y": bin_y,
        "bin_p": bin_p,
    }


def _weighted_cluster_metric_draws(
    frame: pd.DataFrame,
    cluster_counts: np.ndarray,
    cluster_order: np.ndarray,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Evaluate metrics for supplied cluster-frequency bootstrap draws."""
    result = {
        metric: np.full(len(cluster_counts), np.nan, dtype=float)
        for metric in METRICS
    }
    if frame.empty or len(cluster_order) == 0 or len(cluster_counts) == 0:
        return result
    components = _cluster_metric_components(frame, cluster_order)
    sorted_y = components["sorted_y"]
    sorted_cluster = components["sorted_cluster"].astype(np.int64)
    score_starts = components["score_starts"].astype(np.int64)

    for start in range(0, len(cluster_counts), batch_size):
        stop = min(start + batch_size, len(cluster_counts))
        counts = cluster_counts[start:stop].astype(float, copy=False)
        row_weights = counts[:, sorted_cluster]
        success_rows = row_weights * sorted_y
        failure_rows = row_weights - success_rows
        success = np.add.reduceat(success_rows, score_starts, axis=1)
        failure = np.add.reduceat(failure_rows, score_starts, axis=1)
        n_success = success.sum(axis=1)
        n_failure = failure.sum(axis=1)
        valid = (n_success > 0) & (n_failure > 0)

        # Scores are ascending.  The weighted Mann-Whitney form handles ties
        # identically to roc_auc_score.
        failure_below = np.cumsum(failure, axis=1) - failure
        auc_numerator = np.sum(
            success * (failure_below + 0.5 * failure), axis=1
        )
        np.divide(
            auc_numerator,
            n_success * n_failure,
            out=result["auc"][start:stop],
            where=valid,
        )

        # Failure risk is 1-p, so ascending success probability is descending
        # risk.  Grouping tied scores reproduces average_precision_score.
        total = success + failure
        cumulative_failure = np.cumsum(failure, axis=1)
        cumulative_total = np.cumsum(total, axis=1)
        precision = np.divide(
            cumulative_failure,
            cumulative_total,
            out=np.zeros_like(cumulative_failure),
            where=cumulative_total > 0,
        )
        ap_numerator = np.sum(precision * failure, axis=1)
        np.divide(
            ap_numerator,
            n_failure,
            out=result["pr_auc_risk"][start:stop],
            where=n_failure > 0,
        )

        total_n = counts @ components["cluster_n"]
        brier_sum = counts @ components["squared_error"]
        np.divide(
            brier_sum,
            total_n,
            out=result["brier"][start:stop],
            where=total_n > 0,
        )

        bin_n = counts @ components["bin_n"].T
        bin_y = counts @ components["bin_y"].T
        bin_p = counts @ components["bin_p"].T
        bin_gap = np.divide(
            np.abs(bin_y - bin_p),
            bin_n,
            out=np.zeros_like(bin_n),
            where=bin_n > 0,
        )
        weighted_gap = np.sum(bin_n * bin_gap, axis=1)
        np.divide(
            weighted_gap,
            total_n,
            out=result["ece"][start:stop],
            where=total_n > 0,
        )
        # ``metric_values`` treats a single-class resample as undefined for the
        # whole metric vector; retain that historical/bootstrap behaviour.
        for metric in METRICS:
            result[metric][start:stop][~valid] = np.nan
    return result


def _bootstrap_cluster_counts(
    n_boot: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_boot <= 0 or n_clusters <= 0:
        return np.empty((0, n_clusters), dtype=np.int16)
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    return rng.multinomial(n_clusters, probabilities, size=n_boot)


def cluster_bootstrap_summary(
    predictions: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Point estimates and presentation-cluster bootstrap intervals."""
    rows = []
    keys = ["dataset", "week", "protocol", "model"]
    grouped = predictions.groupby(keys)
    total_cells = grouped.ngroups
    for cell_index, (values, group) in enumerate(grouped, start=1):
        if verbose:
            print(
                f"[bootstrap-summary] {cell_index}/{total_cells} "
                f"week={values[1]} protocol={values[2]} model={values[3]}",
                flush=True,
            )
        point = metric_values(group)
        rng = np.random.default_rng(seed)
        clusters = group["cluster_id"].dropna().astype(str).unique()
        cluster_counts = _bootstrap_cluster_counts(n_boot, len(clusters), rng)
        boot = _weighted_cluster_metric_draws(group, cluster_counts, clusters)
        record = dict(zip(keys, values))
        record.update(
            n=len(group),
            n_clusters=group["cluster_id"].nunique(),
            cluster_unit=(
                group["cluster_unit"].iloc[0]
                if "cluster_unit" in group else "presentation"
            ),
        )
        for metric in METRICS:
            draws = np.asarray(boot[metric], dtype=float)
            draws = draws[np.isfinite(draws)]
            record[f"{metric}_mean"] = point[metric]
            record[f"{metric}_ci_low"] = (
                float(np.quantile(draws, 0.025)) if len(draws) else np.nan
            )
            record[f"{metric}_ci_high"] = (
                float(np.quantile(draws, 0.975)) if len(draws) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def cluster_bootstrap_distortion(
    predictions: pd.DataFrame,
    dataset: str,
    n_boot: int = 1000,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Metric differences against the identified intervention risk set."""
    reference = "cutoff_valid" if dataset == "oulab" else "static_full"
    rows = []
    grouped = predictions.groupby(["week", "model"])
    total_cells = grouped.ngroups
    for cell_index, ((week, model), cell) in enumerate(grouped, start=1):
        ref = cell[cell["protocol"] == reference]
        if ref.empty:
            continue
        for protocol in cell["protocol"].unique():
            if protocol == reference:
                continue
            other = cell[cell["protocol"] == protocol]
            union = pd.concat([ref, other], ignore_index=True)
            if verbose:
                print(
                    f"[bootstrap-distortion] {cell_index}/{total_cells} "
                    f"week={week} model={model} protocol={protocol}",
                    flush=True,
                )
            point_ref = metric_values(ref)
            point_other = metric_values(other)
            rng = np.random.default_rng(seed)
            clusters = union["cluster_id"].dropna().astype(str).unique()
            cluster_counts = _bootstrap_cluster_counts(n_boot, len(clusters), rng)
            ref_draws = _weighted_cluster_metric_draws(
                ref, cluster_counts, clusters
            )
            other_draws = _weighted_cluster_metric_draws(
                other, cluster_counts, clusters
            )
            for metric in METRICS:
                values = other_draws[metric] - ref_draws[metric]
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "week": week,
                        "model": model,
                        "protocol": protocol,
                        "reference_protocol": reference,
                        "metric": metric,
                        "delta": point_other[metric] - point_ref[metric],
                        "ci_low": (
                            float(np.quantile(values, 0.025))
                            if len(values) else np.nan
                        ),
                        "ci_high": (
                            float(np.quantile(values, 0.975))
                            if len(values) else np.nan
                        ),
                        "n_clusters": union["cluster_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


_AA = "activity_conditioned"
_VV = "cutoff_valid"
_AV = CROSS_PROTOCOL
_VA = CROSS_PROTOCOL_VA

# Each contrast is a linear combination of cell metrics.  Two-term entries keep
# the historical ``left``/``right`` reporting; the symmetric entries average the
# two traversals of the 2x2 design and are therefore four-term combinations.
#
#   path 1 (evaluation first): AA-VV = (AA-AV) + (AV-VV)
#   path 2 (training   first): AA-VV = (AA-VA) + (VA-VV)
#   symmetric (Shapley):       AA-VV = eval_shapley + train_shapley
#
# All three sum to the same joint difference, so the symmetric split is exact
# and, unlike either single path, does not depend on traversal order.
DECOMPOSITION_CONTRASTS = (
    ("joint_population_difference", {_AA: 1.0, _VV: -1.0}),
    ("evaluation_population_component", {_AA: 1.0, _AV: -1.0}),
    ("training_population_component", {_AV: 1.0, _VV: -1.0}),
    ("training_population_component_at_activity", {_AA: 1.0, _VA: -1.0}),
    ("evaluation_population_component_at_valid", {_VA: 1.0, _VV: -1.0}),
    (
        "evaluation_population_shapley",
        {_AA: 0.5, _AV: -0.5, _VA: 0.5, _VV: -0.5},
    ),
    (
        "training_population_shapley",
        {_AA: 0.5, _VA: -0.5, _AV: 0.5, _VV: -0.5},
    ),
)


def cluster_bootstrap_decomposition(
    predictions: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Decompose the A->A versus V->V metric difference for OULAD.

    For every metric, the point estimates obey the exact identity

    ``M(A,A)-M(V,V) = [M(A,A)-M(A,V)] + [M(A,V)-M(V,V)]``.

    The same cluster-frequency draws are reused for all three cells, retaining
    covariance across contrasts rather than treating overlapping predictions
    or CV folds as independent observations.
    """
    rows: list[dict[str, object]] = []
    required = {_AA, _AV, _VA, _VV}
    grouped = predictions.groupby(["dataset", "week", "model"])
    total_cells = grouped.ngroups
    for cell_index, ((dataset, week, model), cell) in enumerate(grouped, start=1):
        if dataset != "oulab" or not required.issubset(set(cell["protocol"])):
            continue
        if verbose:
            print(
                f"[bootstrap-decomposition] {cell_index}/{total_cells} "
                f"week={week} model={model}",
                flush=True,
            )
        clusters = cell["cluster_id"].dropna().astype(str).unique()
        rng = np.random.default_rng(seed)
        cluster_counts = _bootstrap_cluster_counts(n_boot, len(clusters), rng)
        points: dict[str, dict[str, float]] = {}
        draws: dict[str, dict[str, np.ndarray]] = {}
        for protocol in required:
            protocol_frame = cell[cell["protocol"].eq(protocol)]
            points[protocol] = metric_values(protocol_frame)
            draws[protocol] = _weighted_cluster_metric_draws(
                protocol_frame, cluster_counts, clusters
            )

        def combine_points(weights: dict[str, float], metric: str) -> float:
            return float(
                sum(coef * points[cell][metric] for cell, coef in weights.items())
            )

        def combine_draws(weights: dict[str, float], metric: str) -> np.ndarray:
            total = np.zeros_like(next(iter(draws.values()))[metric])
            for cell, coef in weights.items():
                total = total + coef * draws[cell][metric]
            return total

        # The symmetric split must reproduce the joint difference exactly; a
        # nonzero residual would mean the four cells were not scored on the
        # same folds.
        identity = {
            metric: (
                combine_points({_AA: 1.0, _VV: -1.0}, metric)
                - combine_points(
                    {_AA: 0.5, _AV: -0.5, _VA: 0.5, _VV: -0.5}, metric
                )
                - combine_points(
                    {_AA: 0.5, _VA: -0.5, _AV: 0.5, _VV: -0.5}, metric
                )
            )
            for metric in METRICS
        }
        for contrast, weights in DECOMPOSITION_CONTRASTS:
            positive = [cell for cell, coef in weights.items() if coef > 0]
            negative = [cell for cell, coef in weights.items() if coef < 0]
            for metric in METRICS:
                values = combine_draws(weights, metric)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "week": week,
                        "model": model,
                        "contrast": contrast,
                        "left_protocol": (
                            positive[0] if len(positive) == 1 else "combination"
                        ),
                        "right_protocol": (
                            negative[0] if len(negative) == 1 else "combination"
                        ),
                        "metric": metric,
                        "delta": combine_points(weights, metric),
                        "ci_low": (
                            float(np.quantile(values, 0.025))
                            if len(values) else np.nan
                        ),
                        "ci_high": (
                            float(np.quantile(values, 0.975))
                            if len(values) else np.nan
                        ),
                        "n_clusters": len(clusters),
                        "cluster_unit": (
                            cell["cluster_unit"].iloc[0]
                            if "cluster_unit" in cell else "presentation"
                        ),
                        "identity_residual": identity[metric],
                    }
                )
    return pd.DataFrame(rows)


def risk_set_stratum_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe valid-active and valid-silent performance by training arm."""
    frame = predictions[
        predictions["protocol"].isin((CROSS_PROTOCOL, "cutoff_valid"))
        & predictions["membership_class"].isin(
            ("eligible_active", "eligible_silent")
        )
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["dataset", "week", "model", "protocol", "membership_class"]
    ):
        dataset, week, model, protocol, stratum = keys
        metrics = metric_values(group)
        rows.append(
            {
                "dataset": dataset,
                "week": week,
                "model": model,
                "protocol": protocol,
                "train_protocol": group["train_protocol"].iloc[0],
                "eval_protocol": group["eval_protocol"].iloc[0],
                "membership_class": stratum,
                "n": len(group),
                "n_clusters": group["cluster_id"].nunique(),
                "risk_rate": group["risk"].mean(),
                "mean_predicted_risk": group["risk_score"].mean(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _capacity_by_cluster(
    reference: pd.DataFrame, budget: float
) -> dict[str, int]:
    return {
        str(cluster): max(1, int(round(budget * len(group))))
        for cluster, group in reference.groupby("cluster_id")
    }


def select_at_budget(
    scored: pd.DataFrame,
    capacities: dict[str, int],
) -> pd.DataFrame:
    selected = []
    for cluster, group in scored.groupby("cluster_id"):
        k = min(len(group), capacities.get(str(cluster), 0))
        if k > 0:
            selected.append(group.nlargest(k, "risk_score"))
    return pd.concat(selected, ignore_index=True) if selected else scored.iloc[0:0]


def budget_and_coverage_table(
    predictions: pd.DataFrame,
    memberships: pd.DataFrame,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Budget performance plus recall adjusted for unscored eligible learners."""
    rows = []
    selection_rows = []
    for (dataset, week, model), cell in predictions.groupby(
        ["dataset", "week", "model"]
    ):
        reference_protocol = "cutoff_valid" if dataset == "oulab" else "static_full"
        reference = cell[cell["protocol"] == reference_protocol]
        if reference.empty:
            continue
        reference_risks = float(reference["risk"].sum())
        reference_ids = set(map(tuple, reference[KEY].to_numpy()))
        for budget in budgets:
            capacities = _capacity_by_cluster(reference, budget)
            for protocol, group in cell.groupby("protocol"):
                selected = select_at_budget(group, capacities)
                selected_ids = selected[KEY].apply(tuple, axis=1)
                selected_actionable = selected[selected_ids.isin(reference_ids)]
                candidate_ids = set(map(tuple, group[KEY].to_numpy()))
                caught = float(selected["risk"].sum())
                actionable_caught = float(selected_actionable["risk"].sum())
                flagged = len(selected)
                actionable_flagged = len(selected_actionable)
                rows.append(
                    {
                        "dataset": dataset,
                        "week": week,
                        "model": model,
                        "protocol": protocol,
                        "reference_protocol": reference_protocol,
                        "budget": budget,
                        "n_candidates": len(group),
                        "n_reference": len(reference),
                        "n_flagged": flagged,
                        "n_actionable_candidates": len(candidate_ids & reference_ids),
                        "n_actionable_flagged": actionable_flagged,
                        "n_invalid_flagged": flagged - actionable_flagged,
                        "precision": caught / flagged if flagged else np.nan,
                        "actionable_precision": (
                            actionable_caught / flagged if flagged else np.nan
                        ),
                        "precision_among_actionable": (
                            actionable_caught / actionable_flagged
                            if actionable_flagged else np.nan
                        ),
                        "wasted_budget_rate": (
                            (flagged - actionable_flagged) / flagged
                            if flagged else np.nan
                        ),
                        "recall_protocol": (
                            caught / float(group["risk"].sum())
                            if group["risk"].sum() > 0 else np.nan
                        ),
                        "coverage_adjusted_recall": (
                            actionable_caught / reference_risks
                            if reference_risks > 0 else np.nan
                        ),
                        "candidate_coverage": (
                            len(candidate_ids & reference_ids) / len(reference)
                        ),
                    }
                )
                if flagged:
                    keep = selected[KEY + ["cluster_id", "risk", "risk_score"]].copy()
                    keep["dataset"] = dataset
                    keep["week"] = week
                    keep["model"] = model
                    keep["protocol"] = protocol
                    keep["budget"] = budget
                    selection_rows.append(keep)
    return pd.DataFrame(rows), (
        pd.concat(selection_rows, ignore_index=True)
        if selection_rows else pd.DataFrame()
    )


def decision_overlap_table(selections: pd.DataFrame) -> pd.DataFrame:
    """Protocol and consecutive-landmark top-k set stability."""
    if selections.empty:
        return pd.DataFrame()
    rows = []
    id_cols = KEY
    for (dataset, week, model, budget), cell in selections.groupby(
        ["dataset", "week", "model", "budget"]
    ):
        reference = "cutoff_valid" if dataset == "oulab" else "static_full"
        ref = cell[cell["protocol"] == reference]
        ref_ids = set(map(tuple, ref[id_cols].to_numpy()))
        for protocol in cell["protocol"].unique():
            if protocol == reference:
                continue
            other_ids = set(
                map(tuple, cell[cell["protocol"] == protocol][id_cols].to_numpy())
            )
            union = ref_ids | other_ids
            rows.append(
                {
                    "comparison_type": "protocol",
                    "dataset": dataset,
                    "week_from": week,
                    "week_to": week,
                    "model": model,
                    "budget": budget,
                    "protocol": protocol,
                    "reference_protocol": reference,
                    "jaccard": len(ref_ids & other_ids) / len(union) if union else np.nan,
                    "intersection": len(ref_ids & other_ids),
                    "union": len(union),
                }
            )

        # Holding the fitted activity-conditioned model fixed, compare its
        # selected set under the activity-only candidate universe (A->A) with
        # the intervention-valid universe (A->V).  This is the decision-level
        # analogue of the evaluation-population component in the metric
        # decomposition.
        # Both traversals of the 2x2 design are reported.  Jaccard agreement is
        # not additive, so these are parallel path comparisons rather than an
        # exact decomposition.
        if dataset == "oulab":
            fixed_model_pairs = (
                # activity-trained estimator held fixed, candidate universe varied
                ("activity_conditioned", CROSS_PROTOCOL),
                # valid-trained estimator held fixed, candidate universe varied
                ("cutoff_valid", CROSS_PROTOCOL_VA),
            )
            for left, right in fixed_model_pairs:
                left_frame = cell[cell["protocol"].eq(left)]
                right_frame = cell[cell["protocol"].eq(right)]
                if left_frame.empty or right_frame.empty:
                    continue
                left_ids = set(map(tuple, left_frame[id_cols].to_numpy()))
                right_ids = set(map(tuple, right_frame[id_cols].to_numpy()))
                union = left_ids | right_ids
                rows.append(
                    {
                        "comparison_type": "evaluation_population",
                        "dataset": dataset,
                        "week_from": week,
                        "week_to": week,
                        "model": model,
                        "budget": budget,
                        "protocol": left,
                        "reference_protocol": right,
                        "jaccard": (
                            len(left_ids & right_ids) / len(union)
                            if union else np.nan
                        ),
                        "intersection": len(left_ids & right_ids),
                        "union": len(union),
                    }
                )

    for (dataset, protocol, model, budget), cell in selections.groupby(
        ["dataset", "protocol", "model", "budget"]
    ):
        weeks = sorted(cell["week"].unique())
        for earlier, later in zip(weeks[:-1], weeks[1:]):
            left = set(
                map(tuple, cell[cell["week"] == earlier][id_cols].to_numpy())
            )
            right = set(
                map(tuple, cell[cell["week"] == later][id_cols].to_numpy())
            )
            union = left | right
            rows.append(
                {
                    "comparison_type": "time",
                    "dataset": dataset,
                    "week_from": earlier,
                    "week_to": later,
                    "model": model,
                    "budget": budget,
                    "protocol": protocol,
                    "reference_protocol": protocol,
                    "jaccard": len(left & right) / len(union) if union else np.nan,
                    "intersection": len(left & right),
                    "union": len(union),
                }
            )
    return pd.DataFrame(rows)


def model_ranking_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank agreement and explicit ranking reversals by protocol."""
    rows = []
    if summary.empty:
        return pd.DataFrame()
    reference_by_dataset = {"oulab": "cutoff_valid", "kdd": "static_full"}
    for (dataset, week), cell in summary.groupby(["dataset", "week"]):
        reference = reference_by_dataset[dataset]
        ref = cell[cell["protocol"] == reference]
        if ref.empty:
            continue
        for protocol in cell["protocol"].unique():
            if protocol == reference:
                continue
            other = cell[cell["protocol"] == protocol]
            for metric, ascending in (
                ("auc_mean", False),
                ("pr_auc_risk_mean", False),
                ("brier_mean", True),
                ("ece_mean", True),
            ):
                joined = ref[["model", metric]].merge(
                    other[["model", metric]], on="model", suffixes=("_ref", "_other")
                )
                if len(joined) < 2:
                    continue
                rank_ref = joined[f"{metric}_ref"].rank(
                    ascending=ascending, method="average"
                )
                rank_other = joined[f"{metric}_other"].rank(
                    ascending=ascending, method="average"
                )
                correlation = float(rank_ref.corr(rank_other, method="pearson"))
                best_ref = joined.loc[
                    joined[f"{metric}_ref"].idxmin()
                    if ascending else joined[f"{metric}_ref"].idxmax(),
                    "model",
                ]
                best_other = joined.loc[
                    joined[f"{metric}_other"].idxmin()
                    if ascending else joined[f"{metric}_other"].idxmax(),
                    "model",
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "week": week,
                        "protocol": protocol,
                        "reference_protocol": reference,
                        "metric": metric.removesuffix("_mean"),
                        "spearman_rank_correlation": correlation,
                        "best_reference": best_ref,
                        "best_protocol": best_other,
                        "ranking_reversal": best_ref != best_other,
                    }
                )
    return pd.DataFrame(rows)


def subgroup_tables(
    dataset: str,
    memberships: pd.DataFrame,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subgroup cohort coverage and budget-level allocation performance."""
    if dataset != "oulab":
        return pd.DataFrame(), pd.DataFrame()
    attributes = ["imd_band", "disability", "gender", "age_band", "highest_education"]
    info = pd.read_csv(get_data_path("raw/studentInfo.csv"))[KEY + attributes]
    membership = memberships.merge(info, on=KEY, how="left")
    coverage_rows = []
    for week, cell in membership.groupby("week"):
        reference = cell[cell["cutoff_valid"]]
        for attribute in attributes:
            for value, ref_group in reference.groupby(attribute, dropna=False):
                ref_ids = set(map(tuple, ref_group[KEY].to_numpy()))
                for protocol in _protocols_for_dataset(dataset):
                    protocol_group = cell[cell[protocol]]
                    protocol_ids = set(map(tuple, protocol_group[KEY].to_numpy()))
                    included = ref_ids & protocol_ids
                    included_frame = ref_group[
                        ref_group[KEY].apply(tuple, axis=1).isin(included)
                    ]
                    coverage_rows.append(
                        {
                            "dataset": dataset,
                            "week": week,
                            "attribute": attribute,
                            "group": str(value),
                            "protocol": protocol,
                            "n_reference": len(ref_group),
                            "n_included": len(included),
                            "coverage": len(included) / len(ref_group),
                            "reference_risk_rate": ref_group["risk"].mean(),
                            "included_risk_rate": included_frame["risk"].mean(),
                        }
                    )

    # Evaluate subgroup decisions on the intervention-valid population. Invalid
    # selections consume budget but contribute zero actionable benefit.
    performance_rows = []
    selection_keys = ["dataset", "week", "model", "protocol", "budget"]
    for keys, selected in selections.groupby(selection_keys):
        dataset_value, week, model, protocol, budget = keys
        reference = membership[
            membership["week"].eq(week) & membership["cutoff_valid"]
        ].copy()
        candidate = predictions[
            predictions["week"].eq(week)
            & predictions["model"].eq(model)
            & predictions["protocol"].eq(protocol)
        ]
        candidate_ids = set(map(tuple, candidate[KEY].to_numpy()))
        selected_ids = set(map(tuple, selected[KEY].to_numpy()))
        selected_info = selected.merge(info, on=KEY, how="left")
        for attribute in attributes:
            reference_group = reference.assign(
                _subgroup=reference[attribute].astype(str)
            )
            selected_group = selected_info.assign(
                _subgroup=selected_info[attribute].astype(str)
            )
            for value, subgroup in reference_group.groupby("_subgroup"):
                subgroup_ids = set(map(tuple, subgroup[KEY].to_numpy()))
                valid_selected_ids = subgroup_ids & selected_ids
                selected_total = int(selected_group["_subgroup"].eq(value).sum())
                selected_valid = subgroup[
                    subgroup[KEY].apply(tuple, axis=1).isin(valid_selected_ids)
                ]
                risks = float(subgroup["risk"].sum())
                caught = float(selected_valid["risk"].sum())
                n_candidates = len(subgroup_ids & candidate_ids)
                performance_rows.append(
                    {
                        "dataset": dataset_value,
                        "week": week,
                        "model": model,
                        "protocol": protocol,
                        "budget": budget,
                        "attribute": attribute,
                        "group": value,
                        "n_reference": len(subgroup),
                        "n_candidates": n_candidates,
                        "n_selected": selected_total,
                        "n_actionable_selected": len(selected_valid),
                        "candidate_coverage": n_candidates / len(subgroup),
                        "selection_rate": len(selected_valid) / len(subgroup),
                        "actionable_precision": (
                            caught / selected_total if selected_total else np.nan
                        ),
                        "coverage_adjusted_risk_recall": (
                            caught / risks if risks > 0 else np.nan
                        ),
                        "wasted_budget_rate": (
                            (selected_total - len(selected_valid)) / selected_total
                            if selected_total else np.nan
                        ),
                    }
                )
    return pd.DataFrame(coverage_rows), pd.DataFrame(performance_rows)


def excluded_profile_tables(
    dataset: str, memberships: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe eligible silent learners against eligible active learners."""
    if dataset != "oulab":
        return pd.DataFrame(), pd.DataFrame()
    info = pd.read_csv(get_data_path("raw/studentInfo.csv"))
    registration = pd.read_csv(get_data_path("raw/studentRegistration.csv"))
    frame = memberships.merge(info, on=KEY, how="left").merge(
        registration, on=KEY, how="left", suffixes=("", "_registration")
    )
    categorical = [
        "gender", "region", "highest_education", "imd_band", "age_band", "disability"
    ]
    numeric = ["num_of_prev_attempts", "studied_credits", "date_registration"]
    categorical_rows = []
    numeric_rows = []
    for week, cell in frame.groupby("week"):
        valid = cell[cell["cutoff_valid"]].copy()
        valid["observability"] = np.where(
            valid["has_activity"], "eligible_active", "eligible_silent"
        )
        for attribute in categorical:
            counts = (
                valid.groupby(["observability", attribute], dropna=False)
                .agg(n=("label", "size"), risk_rate=("risk", "mean"))
                .reset_index()
            )
            totals = valid.groupby("observability").size().to_dict()
            for _, row in counts.iterrows():
                categorical_rows.append(
                    {
                        "dataset": dataset,
                        "week": week,
                        "attribute": attribute,
                        "group": str(row[attribute]),
                        "observability": row["observability"],
                        "n": int(row["n"]),
                        "share": row["n"] / totals[row["observability"]],
                        "risk_rate": row["risk_rate"],
                    }
                )
        for attribute in numeric:
            values = pd.to_numeric(valid[attribute], errors="coerce")
            valid_numeric = valid.assign(_value=values)
            active = valid_numeric.loc[valid_numeric["has_activity"], "_value"].dropna()
            silent = valid_numeric.loc[~valid_numeric["has_activity"], "_value"].dropna()
            pooled_sd = math.sqrt(
                (
                    max(len(active) - 1, 0) * active.var(ddof=1)
                    + max(len(silent) - 1, 0) * silent.var(ddof=1)
                )
                / max(len(active) + len(silent) - 2, 1)
            )
            numeric_rows.append(
                {
                    "dataset": dataset,
                    "week": week,
                    "attribute": attribute,
                    "active_n": len(active),
                    "silent_n": len(silent),
                    "active_mean": active.mean(),
                    "silent_mean": silent.mean(),
                    "standardized_mean_difference": (
                        (silent.mean() - active.mean()) / pooled_sd
                        if pooled_sd > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows)


def _hazard_frame_for_model(
    config: BenchmarkConfig,
    model_name: str,
) -> pd.DataFrame:
    """Build a union-column person-week table for one representation."""
    if config.dataset != "oulab":
        return pd.DataFrame()
    roster = load_roster(config.dataset)
    frames = []
    feature_union: set[str] = set()
    for week in config.weeks:
        events = _load_weekly_events(week)
        membership = cohort_membership(config.dataset, week, roster, events)
        representation = _representations(
            events, week, seed=config.seed, models=(model_name,)
        )[model_name]
        aligned, feature_cols = align_features_to_roster(
            config.dataset, week, roster, representation
        )
        feature_union.update(feature_cols)
        frame = membership[membership["cutoff_valid"]].merge(
            aligned.drop(columns=["no_activity"], errors="ignore"),
            on=KEY,
            how="left",
        )
        cutoff = 7 * week
        event_time = pd.to_numeric(frame["date_unregistration"], errors="coerce")
        frame["hazard_event"] = (
            event_time.notna()
            & (event_time >= cutoff)
            & (event_time < cutoff + config.hazard_days)
        ).astype(int)
        frame["landmark_week"] = week
        frame["_split_group"] = _group_id(frame, "presentation")
        frame["cluster_id"] = _group_id(frame, config.cluster)
        frame["cluster_unit"] = config.cluster
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    features = sorted(feature_union | {"landmark_week"})
    all_rows[features] = (
        all_rows[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    all_rows.attrs["features"] = features
    return all_rows


def run_discrete_hazard(config: BenchmarkConfig, verbose: bool = False) -> pd.DataFrame:
    """Pooled person-interval HGB predicting withdrawal within ``hazard_days``."""
    if config.dataset != "oulab":
        return pd.DataFrame()
    roster = load_roster(config.dataset)
    fold_maps = make_fold_maps(
        roster, config.folds, config.repeats, config.seed, config.split_unit
    )
    rows = []
    for model_name in config.models:
        frame = _hazard_frame_for_model(config, model_name)
        if frame.empty:
            continue
        feature_cols = frame.attrs["features"]
        if verbose:
            interval_name = (
                "person-weeks"
                if config.hazard_days == 7
                else f"person-{config.hazard_days}-day intervals"
            )
            print(
                f"[hazard] model={model_name}: {len(frame):,} {interval_name}, "
                f"event rate={frame['hazard_event'].mean():.3f}",
                flush=True,
            )
        for repeat, mapping in fold_maps.items():
            frame["_fold"] = frame["_split_group"].map(mapping)
            for fold in range(1, config.folds + 1):
                test = frame["_fold"].eq(fold)
                train = frame["_fold"].ne(fold)
                y_train = frame.loc[train, "hazard_event"]
                y_test = frame.loc[test, "hazard_event"]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                estimator = _model(
                    model_name, config.seed + 5000 + repeat * 100 + fold
                )
                estimator.fit(frame.loc[train, feature_cols], y_train)
                p_event = estimator.predict_proba(frame.loc[test, feature_cols])[:, 1]
                record = frame.loc[
                    test,
                    KEY
                    + [
                        "landmark_week",
                        "hazard_event",
                        "cluster_id",
                        "cluster_unit",
                    ],
                ].copy()
                record["dataset"] = config.dataset
                record["model"] = model_name
                record["repeat"] = repeat
                record["fold"] = fold
                record["p_event"] = p_event
                rows.append(record)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_discrete_hazard(
    predictions: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    averaged = (
        predictions.groupby(
            [
                "dataset", "model", "landmark_week", *KEY,
                "hazard_event", "cluster_id", "cluster_unit",
            ],
            as_index=False,
        )
        .agg(p_event=("p_event", "mean"))
    )
    proxy = averaged.rename(
        columns={
            "landmark_week": "week",
            "hazard_event": "risk",
            "p_event": "risk_score",
        }
    )
    proxy["y"] = 1 - proxy["risk"]
    proxy["p_success"] = 1 - proxy["risk_score"]
    proxy["protocol"] = "discrete_hazard"
    return cluster_bootstrap_summary(proxy, n_boot=n_boot, seed=seed)


def _flow_patch(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0a: float,
    y0b: float,
    y1a: float,
    y1b: float,
    color: str,
    alpha: float = 0.45,
) -> None:
    c = 0.45 * (x1 - x0)
    vertices = [
        (x0, y0a), (x0 + c, y0a), (x1 - c, y1a), (x1, y1a),
        (x1, y1b), (x1 - c, y1b), (x0 + c, y0b), (x0, y0b), (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), color=color, alpha=alpha, lw=0))


def plot_cohort_flow(
    membership: pd.DataFrame, output: Path, week: int | None = None
) -> Path:
    """Static Sankey-style flow for the most policy-relevant OULAD landmark."""
    week = int(week if week is not None else membership["week"].min())
    frame = membership[membership["week"] == week].copy()
    # A handful of OULAD click records precede the recorded registration date.
    # Preserve those data-quality anomalies as their own flows so the activity
    # side of the Sankey reconciles exactly with the evaluated cohort.
    frame["flow_class"] = frame["membership_class"]
    pre_registration = frame["membership_class"].eq("not_yet_registered")
    frame.loc[pre_registration, "flow_class"] = np.where(
        frame.loc[pre_registration, "has_activity"],
        "not_yet_registered_active",
        "not_yet_registered_silent",
    )
    counts = frame["flow_class"].value_counts().to_dict()
    classes = [
        "eligible_active", "eligible_silent", "outcome_realized_active",
        "outcome_realized_silent", "not_yet_registered_active",
        "not_yet_registered_silent",
    ]
    colors = {
        "eligible_active": "#4c78a8",
        "eligible_silent": "#f2cf5b",
        "outcome_realized_active": "#e45756",
        "outcome_realized_silent": "#b279a2",
        "not_yet_registered_active": "#777777",
        "not_yet_registered_silent": "#bdbdbd",
    }
    total = max(len(frame), 1)
    gap = 0.012
    heights = {name: counts.get(name, 0) / total for name in classes}
    scale = (1 - gap * (len(classes) - 1)) / max(sum(heights.values()), 1e-9)
    heights = {name: value * scale for name, value in heights.items()}
    middle = {}
    cursor = 0.0
    for name in classes:
        middle[name] = (cursor, cursor + heights[name])
        cursor += heights[name] + gap

    right_classes = ["included_activity", "excluded_activity"]
    included_names = {
        "eligible_active", "outcome_realized_active",
        "not_yet_registered_active",
    }
    right_counts = {
        "included_activity": sum(counts.get(n, 0) for n in included_names),
        "excluded_activity": len(frame) - sum(counts.get(n, 0) for n in included_names),
    }
    right = {}
    cursor = 0.0
    for name in right_classes:
        source_names = (
            included_names
            if name == "included_activity"
            else set(classes) - included_names
        )
        h = sum(heights[source] for source in source_names)
        right[name] = (cursor, cursor + h)
        cursor += h + gap

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in classes:
        destination = "included_activity" if name in included_names else "excluded_activity"
        y0a, y0b = middle[name]
        y1a, y1b = right[destination]
        h = y0b - y0a
        if destination == "included_activity":
            used = sum(
                heights[n] for n in classes[:classes.index(name)]
                if n in included_names
            )
        else:
            used = sum(
                heights[n] for n in classes[:classes.index(name)]
                if n not in included_names
            )
        y1a = right[destination][0] + used
        y1b = y1a + h
        _flow_patch(ax, 0.08, 0.92, y0a, y0b, y1a, y1b, colors[name])
        ax.add_patch(Rectangle((0.03, y0a), 0.05, h, color=colors[name]))
        if h > 0.018:
            ax.text(
                0.025, (y0a + y0b) / 2,
                f"{name.replace('_', ' ')}\n{counts.get(name, 0):,}",
                ha="right", va="center", fontsize=8,
            )
    for name, (a, b) in right.items():
        color = "#4c78a8" if name == "included_activity" else "#bab0ac"
        ax.add_patch(Rectangle((0.92, a), 0.05, b - a, color=color))
        ax.text(
            0.975, (a + b) / 2,
            f"{name.replace('_', ' ')}\n{right_counts[name]:,}",
            ha="left", va="center", fontsize=9,
        )
    ax.set_xlim(-0.28, 1.25)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        f"Cohort exchange at week {week}: event-conditioned inclusion",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_population_trajectory(indices: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frame = indices.sort_values("week")
    is_kdd = frame["dataset"].eq("kdd").all()
    comparison_label = (
        "Full labelled roster (event time unavailable)"
        if is_kdd else "Cutoff-valid risk set"
    )
    exclusion_label = (
        "Silent exclusion (eligibility not identifiable)"
        if is_kdd else "Eligible-silent exclusion"
    )
    axes[0].plot(frame["week"], frame["n_activity"], marker="o", label="Activity-conditioned")
    axes[0].plot(
        frame["week"], frame["n_cutoff_valid"], marker="o",
        label=comparison_label,
    )
    axes[0].set_ylabel("Learner-presentations")
    axes[0].set_xlabel("Snapshot week")
    axes[0].set_title("Population size")
    axes[0].legend(frameon=False)
    axes[1].plot(
        frame["week"], frame["activity_contamination_rate"],
        marker="o", label="Known-outcome contamination",
    )
    axes[1].plot(
        frame["week"], frame["silent_exclusion_rate"],
        marker="o", label=exclusion_label,
    )
    axes[1].set_ylabel("Share")
    axes[1].set_xlabel("Snapshot week")
    axes[1].set_title("Cohort-validity errors")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_metric_distortion(distortion: pd.DataFrame, output: Path) -> Path:
    frame = distortion[distortion["metric"].isin(["auc", "pr_auc_risk"])].copy()
    if frame.empty:
        return output
    metrics = ["auc", "pr_auc_risk"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, metrics):
        subset = frame[frame["metric"] == metric]
        for (protocol, model), group in subset.groupby(["protocol", "model"]):
            group = group.sort_values("week")
            ax.plot(
                group["week"], group["delta"], marker="o",
                label=f"{protocol}: {model}",
            )
            ax.fill_between(
                group["week"], group["ci_low"], group["ci_high"], alpha=0.10
            )
        ax.axhline(0, color="black", lw=1)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("Snapshot week")
        ax.set_ylabel("Protocol − valid-risk-set")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_decision_distortion(overlap: pd.DataFrame, output: Path) -> Path:
    frame = overlap[
        (overlap["comparison_type"] == "protocol")
        & np.isclose(overlap["budget"], 0.05)
    ]
    if frame.empty:
        return output
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (protocol, model), group in frame.groupby(["protocol", "model"]):
        group = group.sort_values("week_from")
        ax.plot(
            group["week_from"], group["jaccard"], marker="o",
            label=f"{protocol}: {model}",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Snapshot week")
    ax.set_ylabel("Top-5% Jaccard with valid-risk-set policy")
    ax.set_title("Decision distortion from cohort construction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_equity_distortion(coverage: pd.DataFrame, output: Path) -> Path:
    frame = coverage[
        (coverage["protocol"] == "activity_conditioned")
        & (coverage["attribute"].isin(["imd_band", "disability", "gender"]))
    ].copy()
    if frame.empty:
        return output
    frame["exclusion"] = 1 - frame["coverage"]
    summary = (
        frame.groupby(["week", "attribute"])
        .agg(max_exclusion=("exclusion", "max"), min_exclusion=("exclusion", "min"))
        .reset_index()
    )
    summary["coverage_gap"] = summary["max_exclusion"] - summary["min_exclusion"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for attribute, group in summary.groupby("attribute"):
        ax.plot(
            group["week"], group["coverage_gap"], marker="o",
            label=attribute.replace("_", " "),
        )
    ax.set_xlabel("Snapshot week")
    ax.set_ylabel("Max − min subgroup exclusion rate")
    ax.set_title("Equity distortion before modelling")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def write_protocol(config: BenchmarkConfig, output_dir: Path) -> Path:
    payload = {
        "dataset": config.dataset,
        "weeks": list(config.weeks),
        "models": list(config.models),
        "folds": config.folds,
        "repeats": config.repeats,
        "seed": config.seed,
        "budgets": list(config.budgets),
        "bootstrap_iterations": config.bootstrap_iterations,
        "cluster_inference_unit": config.cluster,
        "fold_parallel_jobs": config.jobs,
        "hazard_days": config.hazard_days,
        "cohort_protocols": list(_protocols_for_dataset(config.dataset)),
        "protocols": list(_prediction_protocols_for_dataset(config.dataset)),
        "train_evaluation_cells": [
            {
                "protocol": output_protocol,
                "train_protocol": train_protocol,
                "eval_protocol": eval_protocol,
            }
            for train_protocol, evaluations in _train_evaluation_specs(
                config.dataset
            ).items()
            for output_protocol, eval_protocol in evaluations
        ],
        "risk_set_note": (
            "registration/withdrawal dates available"
            if config.dataset == "oulab"
            else "withdrawal event time unavailable; static_full is the risk-set proxy"
        ),
    }
    path = output_dir / "protocol.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def reference_baseline_manifest(models: Iterable[str]) -> pd.DataFrame:
    """Return citation and implementation scope for selected paper baselines."""
    rows = []
    for model in models:
        metadata = REFERENCE_BASELINE_METADATA.get(model)
        if metadata is not None:
            rows.append({"model": model, **metadata})
    return pd.DataFrame(rows)


def _checkpoint_signature(
    config: BenchmarkConfig,
    execution_mode: str = "full",
) -> dict[str, object]:
    """Fields that must match before fitted prediction cells can be reused."""
    return {
        "schema": 2,
        "dataset": config.dataset,
        "folds": config.folds,
        "repeats": config.repeats,
        "seed": config.seed,
        "cluster": config.cluster,
        "prediction_protocols": list(
            _prediction_protocols_for_dataset(config.dataset)
        ),
        "execution_mode": execution_mode,
    }


def _prepare_checkpoint_dir(
    config: BenchmarkConfig,
    output_dir: Path,
    resume: bool,
    execution_mode: str = "full",
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    expected = _checkpoint_signature(config, execution_mode)
    if not resume:
        # The user explicitly requested a clean recomputation.  Limit removal
        # to files created by this checkpoint implementation in the resolved
        # dataset-specific checkpoint directory.
        for checkpoint in checkpoint_dir.glob("week_[0-9][0-9][0-9]__*.csv.gz"):
            checkpoint.unlink()
    if resume and manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                "checkpoint settings do not match this run; use --no-resume "
                f"or a new --output directory (found {observed}, expected {expected})"
            )
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return checkpoint_dir


def _completed_run_matches(config: BenchmarkConfig, output_dir: Path) -> bool:
    """Return whether final landmark files can safely seed a resumed run."""
    required = (
        output_dir / "predictions.csv.gz",
        output_dir / "cohort_membership.csv.gz",
        output_dir / "cohort_composition.csv",
        output_dir / "protocol.json",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        protocol = json.loads(required[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "dataset": config.dataset,
        "weeks": list(config.weeks),
        "models": list(config.models),
        "folds": config.folds,
        "repeats": config.repeats,
        "seed": config.seed,
        "cluster_inference_unit": config.cluster,
        "protocols": list(_prediction_protocols_for_dataset(config.dataset)),
    }
    return all(protocol.get(key) == value for key, value in expected.items())


def run_cohort_exchange(
    config: BenchmarkConfig,
    output_root: Path,
    run_hazard: bool = True,
    verbose: bool = False,
    resume: bool = True,
    augment_from: Path | None = None,
    fit_only: bool = False,
) -> dict[str, Path]:
    """End-to-end benchmark and publication artifact writer."""
    output_dir = output_root / config.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if augment_from is not None and config.dataset != "oulab":
        raise ValueError("cross-protocol augmentation is identified only for OULAD")
    execution_mode = (
        "augment_paired_activity_evaluations"
        if augment_from is not None else "full"
    )
    checkpoint_dir = _prepare_checkpoint_dir(
        config, output_dir, resume, execution_mode
    )
    outputs: dict[str, Path] = {}

    def save(name: str, frame: pd.DataFrame) -> Path:
        path = output_dir / name
        _atomic_to_csv(frame, path)
        outputs[name] = path
        return path

    if resume and _completed_run_matches(config, output_dir):
        if verbose:
            print(
                f"[resume-final] {config.dataset}: loading completed landmark "
                "predictions; clustered inference will be regenerated",
                flush=True,
            )
        predictions = pd.read_csv(output_dir / "predictions.csv.gz")
        memberships = pd.read_csv(output_dir / "cohort_membership.csv.gz")
        composition = pd.read_csv(output_dir / "cohort_composition.csv")
        outputs["predictions.csv.gz"] = output_dir / "predictions.csv.gz"
        outputs["cohort_membership.csv.gz"] = output_dir / "cohort_membership.csv.gz"
        outputs["cohort_composition.csv"] = output_dir / "cohort_composition.csv"
    elif augment_from is not None:
        source_dir = Path(augment_from)
        if (source_dir / config.dataset).is_dir():
            source_dir = source_dir / config.dataset
        source_protocol_path = source_dir / "protocol.json"
        source_predictions_path = source_dir / "predictions.csv.gz"
        if not source_protocol_path.exists() or not source_predictions_path.exists():
            raise FileNotFoundError(
                "--augment-from must contain protocol.json and predictions.csv.gz "
                f"(resolved {source_dir})"
            )
        source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
        expected_source = {
            "dataset": config.dataset,
            "weeks": list(config.weeks),
            "models": list(config.models),
            "folds": config.folds,
            "repeats": config.repeats,
            "seed": config.seed,
            "cluster_inference_unit": config.cluster,
        }
        mismatches = {
            key: (source_protocol.get(key), value)
            for key, value in expected_source.items()
            if source_protocol.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "augmentation source settings do not match the requested run: "
                f"{mismatches}"
            )
        if verbose:
            print(
                f"[augment] reusing completed within-protocol predictions from "
                f"{source_dir}",
                flush=True,
            )
        source_predictions = pd.read_csv(source_predictions_path)
        available = set(source_predictions["protocol"].astype(str).unique())
        # Refit only those training arms whose output cells are incomplete in
        # the source run.  Both cells of an arm are always recomputed together
        # so that, for example, V->V and V->A come from the *same* fitted
        # estimator; otherwise the training-population contrast would compare
        # two independently fitted models.
        missing_specs = {
            train_protocol: evaluation_specs
            for train_protocol, evaluation_specs in _train_evaluation_specs(
                config.dataset
            ).items()
            if not {cell for cell, _ in evaluation_specs}.issubset(available)
        }
        if not missing_specs:
            raise ValueError(
                "augmentation source already contains every train/evaluation "
                "cell for this dataset; nothing to augment"
            )
        recomputed = {
            cell
            for evaluation_specs in missing_specs.values()
            for cell, _ in evaluation_specs
        }
        if verbose:
            print(
                f"[augment] refitting {sorted(missing_specs)} to produce "
                f"{sorted(recomputed)}",
                flush=True,
            )
        cross_predictions, memberships, composition = run_landmark_benchmark(
            config,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            fit_specs=missing_specs,
        )
        source_predictions = source_predictions[
            ~source_predictions["protocol"].isin(recomputed)
        ]
        source_predictions = _with_train_eval_protocols(source_predictions)
        predictions = pd.concat(
            [source_predictions, cross_predictions],
            ignore_index=True,
            sort=False,
        )
        save("predictions.csv.gz", predictions)
        save("cohort_membership.csv.gz", memberships)
        save("cohort_composition.csv", composition)
    else:
        predictions, memberships, composition = run_landmark_benchmark(
            config,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
        # Persist the expensive fitted predictions before any inference or
        # figure generation. An interruption cannot lose model fitting.
        save("predictions.csv.gz", predictions)
        save("cohort_membership.csv.gz", memberships)
        save("cohort_composition.csv", composition)

    if fit_only:
        outputs["protocol.json"] = write_protocol(config, output_dir)
        return outputs

    summary = cluster_bootstrap_summary(
        predictions,
        config.bootstrap_iterations,
        config.seed,
        verbose=verbose,
    )
    save("clustered_summary.csv", summary)
    distortion = cluster_bootstrap_distortion(
        predictions,
        config.dataset,
        config.bootstrap_iterations,
        config.seed,
        verbose=verbose,
    )
    save("metric_distortion.csv", distortion)
    if config.dataset == "oulab":
        decomposition = cluster_bootstrap_decomposition(
            predictions,
            config.bootstrap_iterations,
            config.seed,
            verbose=verbose,
        )
        save("training_evaluation_decomposition.csv", decomposition)
        module_predictions = predictions.assign(
            cluster_id=predictions["code_module"].astype(str),
            cluster_unit="module",
        )
        module_decomposition = cluster_bootstrap_decomposition(
            module_predictions,
            config.bootstrap_iterations,
            config.seed,
            verbose=verbose,
        )
        save(
            "training_evaluation_decomposition_module.csv",
            module_decomposition,
        )
        save("risk_set_stratum_summary.csv", risk_set_stratum_summary(predictions))
    budget, selections = budget_and_coverage_table(
        predictions, memberships, config.budgets
    )
    save("budget_performance.csv", budget)
    overlap = decision_overlap_table(selections)
    save("decision_overlap.csv", overlap)
    rankings = model_ranking_stability(summary)
    save("model_ranking_stability.csv", rankings)
    subgroup_coverage, subgroup_budget = subgroup_tables(
        config.dataset, memberships, predictions, selections
    )
    save("subgroup_coverage.csv", subgroup_coverage)
    save("subgroup_budget.csv", subgroup_budget)
    excluded_categorical, excluded_numeric = excluded_profile_tables(
        config.dataset, memberships
    )
    save("excluded_profile_categorical.csv", excluded_categorical)
    save("excluded_profile_numeric.csv", excluded_numeric)

    baseline_manifest = reference_baseline_manifest(config.models)
    if not baseline_manifest.empty:
        save("reference_baselines.csv", baseline_manifest)

    if run_hazard and config.dataset == "oulab":
        hazard_predictions = run_discrete_hazard(config, verbose=verbose)
        hazard_summary = summarize_discrete_hazard(
            hazard_predictions, config.bootstrap_iterations, config.seed
        )
        save("discrete_hazard_predictions.csv.gz", hazard_predictions)
        save("discrete_hazard_summary.csv", hazard_summary)

    indices = composition[composition["table"] == "indices"].copy()
    outputs["cohort_flow.png"] = plot_cohort_flow(
        memberships, figures / f"cohort_flow_week{min(config.weeks)}.png"
    )
    outputs["population_trajectory.png"] = plot_population_trajectory(
        indices, figures / "population_trajectory.png"
    )
    outputs["metric_distortion.png"] = plot_metric_distortion(
        distortion, figures / "metric_distortion.png"
    )
    outputs["decision_distortion.png"] = plot_decision_distortion(
        overlap, figures / "decision_distortion.png"
    )
    if not subgroup_coverage.empty:
        outputs["equity_distortion.png"] = plot_equity_distortion(
            subgroup_coverage, figures / "equity_distortion.png"
        )
    outputs["protocol.json"] = write_protocol(config, output_dir)
    return outputs
