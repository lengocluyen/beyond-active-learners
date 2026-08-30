"""Repeated group-aware PCG-UT evaluation for OULAD and KDD.

This module intentionally evaluates every representation with the same
HistGradientBoosting risk head.  It therefore separates representation quality
from classifier capacity, unlike comparisons between PCG-UT and a logistic-only
flat baseline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never require a display when plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .paths import get_data_path
from .pcg_ut import KEY, _load_weekly_events, _state_features
from .pcg_ut_graph import build_pcg_ut_graph_features
from .features_static import sensitive_columns
from .traversal import ORDER_COLUMNS


DEFAULT_SNAPSHOTS = {"oulab": [2, 4, 6, 8, 10, 12, 14, 16], "kdd": [1, 2, 3, 4, 5]}
PRIMARY_MODELS = ("flat_hgb", "temporal_hgb", "pcg_ut_no_graph", "pcg_ut")
ALL_MODELS = PRIMARY_MODELS + ("pcg_ut_shuffled_graph", "pcg_ut_no_uncertainty")

# Literature-derived model families available to controlled cohort experiments.
# These are protocol-harmonised adaptations, not claims of exact reproduction:
# every model uses this repository's non-leaky snapshot features and grouped
# folds. The cited papers often use different outcomes, feature windows and
# train/test schemes.
REFERENCE_BASELINES = {
    "oulab": (
        "base_lr_profile",
        "base_lr_clickstream",
        "base_lr_weekly",
        "base_rf_gini",
        "base_rf_entropy",
        "base_svm",
        "base_knn",
        "base_dffnn",
    ),
    "kdd": (
        "base_lr_clickstream",
        "base_lr_weekly",
        "base_rf_gini",
        "base_rf_entropy",
        "base_svm",
        "base_dffnn",
    ),
}

REFERENCE_BASELINE_METADATA = {
    "base_lr_profile": {
        "family": "logistic_regression",
        "features": "enrolment profile",
        "references": "Waheed et al. (2020)",
        "doi": "10.1016/j.chb.2019.106189",
        "scope": "family adaptation",
    },
    "base_lr_clickstream": {
        "family": "logistic_regression",
        "features": "cumulative clickstream",
        "references": "Hassan et al. (2019); Waheed et al. (2020)",
        "doi": "10.1002/int.22129; 10.1016/j.chb.2019.106189",
        "scope": "family adaptation",
    },
    "base_lr_weekly": {
        "family": "logistic_regression",
        "features": "weekly clickstream and assessment sequence",
        "references": "Hassan et al. (2019)",
        "doi": "10.1002/int.22129",
        "scope": "family adaptation",
    },
    "base_rf_gini": {
        "family": "random_forest_gini",
        "features": "non-leaky profile plus weekly activity and assessment",
        "references": "Junejo et al. (2025)",
        "doi": "10.1038/s41598-025-00256-3",
        "scope": "architecture-family transplant",
    },
    "base_rf_entropy": {
        "family": "random_forest_entropy",
        "features": "non-leaky profile plus weekly activity and assessment",
        "references": "Junejo et al. (2025)",
        "doi": "10.1038/s41598-025-00256-3",
        "scope": "architecture-family transplant",
    },
    "base_svm": {
        "family": "linear_svm_calibrated",
        "features": "cumulative clickstream",
        "references": "Waheed et al. (2020)",
        "doi": "10.1016/j.chb.2019.106189",
        "scope": "family adaptation",
    },
    "base_knn": {
        "family": "k_nearest_neighbours",
        "features": "cumulative clickstream",
        "references": "Junejo et al. (2025), literature comparator family",
        "doi": "10.1038/s41598-025-00256-3",
        "scope": "family adaptation",
    },
    "base_dffnn": {
        "family": "deep_feedforward_neural_network",
        "features": "non-leaky profile plus weekly activity and assessment",
        "references": "Waheed et al. (2020); Junejo et al. (2025)",
        "doi": "10.1016/j.chb.2019.106189; 10.1038/s41598-025-00256-3",
        "scope": "architecture-family transplant",
    },
}


def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _flat_features(events: pd.DataFrame, snapshot_week: int) -> pd.DataFrame:
    """Cumulative, non-graph baseline using the same raw event population."""
    key = KEY
    numeric = [col for col in events.columns if col not in key + ["week_index", "competency_id", "competency_id_assessment"]]
    work = events[key].copy()
    for col in numeric:
        work[col] = _numeric(events[col], events.index)
    agg = work.groupby(key, as_index=False)[numeric].sum()
    agg["snapshot_week"] = snapshot_week
    return agg


def _temporal_features(events: pd.DataFrame, snapshot_week: int) -> pd.DataFrame:
    """Weekly event encoding with no graph propagation or graph summaries."""
    cols = ["clicks_total", "active_days", "assess_attempts", "assess_score_mean", "assess_score_max", "assess_score_weighted"]
    work = events[KEY + ["week_index"]].copy()
    for col in cols:
        values = _numeric(events[col] if col in events else None, events.index).clip(lower=0)
        if col == "clicks_total":
            values = np.log1p(values)
        work[col] = values
    pivots = []
    for col in cols:
        p = work.pivot_table(index=KEY, columns="week_index", values=col, aggfunc="mean", fill_value=0.0)
        p = p.reindex(columns=range(1, snapshot_week + 1), fill_value=0.0)
        p.columns = [f"{col}_week{week}" for week in p.columns]
        pivots.append(p)
    out = pd.concat(pivots, axis=1).reset_index()
    out["snapshot_week"] = snapshot_week
    return out


def _reference_all_features(
    events: pd.DataFrame, snapshot_week: int
) -> pd.DataFrame:
    """Reference-paper feature union without withdrawal-derived variables."""
    base = _temporal_features(events, snapshot_week)
    if get_data_path("processed/static_features.csv").exists():
        base = _with_static(base)
    return base


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for value in range(bins):
        mask = bucket == value
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prediction = (p >= 0.5).astype(int)
    risk_y = 1 - y
    risk_prediction = (p < 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc_risk": float(average_precision_score(risk_y, 1.0 - p)),
        "f1_success": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "at_risk_recall": float(recall_score(risk_y, risk_prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": _expected_calibration_error(y, p),
    }


def _representations(
    events: pd.DataFrame, week: int, seed: int = 42, models: tuple[str, ...] | None = None
) -> dict[str, pd.DataFrame]:
    """Build the requested representations, constructing only what is asked for.

    Construction is lazy because the two-level graph states are materially more
    expensive than the path states, and a graph-ablation run rarely needs the
    full model roster.
    """
    builders = {
        "flat_hgb": lambda: _flat_features(events, week),
        "temporal_hgb": lambda: _temporal_features(events, week),
        "pcg_ut_no_graph": lambda: _state_features(events, week, propagation="none"),
        "pcg_ut": lambda: _state_features(events, week, propagation="chain"),
        "pcg_ut_shuffled_graph": lambda: _state_features(events, week, propagation="shuffled", seed=seed),
        "pcg_ut_no_uncertainty": lambda: _state_features(events, week, propagation="chain", include_uncertainty=False),
        # Two-level week x activity-type graph (see src/pcg_ut_graph.py).
        "pcg_g": lambda: _graph(events, week, "full", seed),
        "pcg_g_no_graph": lambda: _graph(events, week, "none", seed),
        "pcg_g_shuffled": lambda: _graph(events, week, "shuffled", seed),
        "pcg_g_temporal_only": lambda: _graph(events, week, "temporal_only", seed),
        "pcg_g_hierarchy_only": lambda: _graph(events, week, "hierarchy_only", seed),
        # Nested ladder: local_only subset no_graph subset residual.
        "pcg_g_local_only": lambda: _graph(events, week, "local_only", seed),
        "pcg_g_residual": lambda: _graph(events, week, "residual", seed),
        "pcg_g_residual_shuffled": lambda: _graph(events, week, "residual_shuffled", seed),
        # Per-learner traversal structure (src/traversal.py). ORDER_COLUMNS are
        # invariant to evidence volume, so a gain from them is attributable to
        # path shape rather than to activity the baselines already encode.
        "trav_order_only": lambda: _with_traversal(
            events[KEY].drop_duplicates().reset_index(drop=True), week, ORDER_COLUMNS
        ),
        "temporal_hgb_trav": lambda: _with_traversal(
            _temporal_features(events, week), week, ORDER_COLUMNS
        ),
        "pcg_g_residual_trav": lambda: _with_traversal(
            _graph(events, week, "residual", seed), week, ORDER_COLUMNS
        ),
        # Tests whether the week-8 deficit is assessment *resolution* rather than
        # anything to do with graph structure.
        "pcg_g_residual_trav_assess": lambda: _with_assessment_weeks(
            _with_traversal(_graph(events, week, "residual", seed), week, ORDER_COLUMNS),
            events,
            week,
        ),
        # --- incremental stack: each rung adds exactly one feature block, so a
        # --- gain is attributable to that block rather than to the bundle.
        "stack_static_only": lambda: _with_static(
            events[KEY].drop_duplicates().reset_index(drop=True)
        ),
        "stack_1_temporal": lambda: _temporal_features(events, week),
        "stack_2_static": lambda: _with_static(_temporal_features(events, week)),
        "stack_3_gap": lambda: _with_behavioral(
            _with_static(_temporal_features(events, week)), week, ("gap",)
        ),
        "stack_4_timing": lambda: _with_behavioral(
            _with_static(_temporal_features(events, week)), week, ("gap", "time")
        ),
        "stack_5_cohort": lambda: _with_behavioral(
            _with_static(_temporal_features(events, week)), week
        ),
        "stack_6_traversal": lambda: _with_traversal(
            _with_behavioral(_with_static(_temporal_features(events, week)), week),
            week,
            ORDER_COLUMNS,
        ),
        # --- protected-attribute ablation -------------------------------------
        # Identical to stack_7_full except for which enrolment attributes are
        # admitted, so the AUC difference is the price of fairness rather than a
        # confound with anything else in the pipeline.
        "fair_all": lambda: _fair_stack(events, week, seed, "all"),
        "fair_core_removed": lambda: _fair_stack(events, week, seed, "core_only_removed"),
        "fair_none": lambda: _fair_stack(events, week, seed, "none"),
        # --- literature baseline families -------------------------------------
        # Each pairs a representation with the estimator that family typically
        # uses (see _ESTIMATORS). Fitted on identical folds and identical
        # snapshot truncation as everything else.
        #
        # profile-only: demographics + enrolment, no behaviour at all
        "base_lr_profile": lambda: _with_static(
            events[KEY].drop_duplicates().reset_index(drop=True)
        ),
        # clickstream aggregates -- the most common OULAD baseline
        "base_lr_clickstream": lambda: _flat_features(events, week),
        # per-week clickstream counts
        "base_lr_weekly": lambda: _temporal_features(events, week),
        "base_rf": lambda: _flat_features(events, week),
        "base_rf_gini": lambda: _reference_all_features(events, week),
        "base_rf_entropy": lambda: _reference_all_features(events, week),
        "base_svm": lambda: _flat_features(events, week),
        "base_knn": lambda: _flat_features(events, week),
        "base_mlp": lambda: _temporal_features(events, week),
        "base_dffnn": lambda: _reference_all_features(events, week),
        "stack_7_full": lambda: _with_traversal(
            _with_behavioral(
                _with_static(
                    _with_assessment_weeks(_graph(events, week, "residual", seed), events, week)
                ),
                week,
            ),
            week,
            ORDER_COLUMNS,
        ),
    }
    wanted = tuple(builders) if models is None else models
    return {name: builders[name]() for name in wanted if name in builders}


def _graph(events: pd.DataFrame, week: int, propagation: str, seed: int) -> pd.DataFrame:
    return build_pcg_ut_graph_features(week, propagation=propagation, seed=seed, events=events)


def _fair_stack(events: pd.DataFrame, week: int, seed: int, sensitive: str) -> pd.DataFrame:
    """The full stack, varying only which protected attributes are admitted."""
    return _with_traversal(
        _with_behavioral(
            _with_static(
                _with_assessment_weeks(_graph(events, week, "residual", seed), events, week),
                sensitive=sensitive,
            ),
            week,
        ),
        week,
        ORDER_COLUMNS,
    )


def _merge_block(base: pd.DataFrame, block: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Left-join a feature block, preserving the base learner population exactly.

    The harness requires every representation in a run to cover the same
    learners, so the join must never add or drop rows. Missing values become
    zeros: a learner absent from the VLE logs has no gap structure, which is
    information rather than missingness.
    """
    keep = [c for c in block.columns if c.split("_")[0] in prefixes]
    merged = base.merge(block[KEY + keep], on=KEY, how="left")
    merged[keep] = merged[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def _with_static(base: pd.DataFrame, sensitive: str = "all") -> pd.DataFrame:
    """Add week-0 enrolment features (see src/features_static.py).

    ``sensitive`` controls which protected attributes are admitted:
    ``"all"`` keeps everything, ``"none"`` drops protected and protected-adjacent
    columns, ``"core_only_removed"`` drops the unambiguously protected ones but
    retains prior attainment. Reporting all three separates "what does accuracy
    cost" from "which attribute is doing the work".
    """
    path = get_data_path("processed/static_features.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_extra_features.py")
    block = pd.read_csv(path)
    if sensitive == "none":
        block = block.drop(columns=sensitive_columns(block, include_adjacent=True))
    elif sensitive == "core_only_removed":
        block = block.drop(columns=sensitive_columns(block, include_adjacent=False))
    elif sensitive != "all":
        raise ValueError(f"unknown sensitive mode: {sensitive}")
    return _merge_block(base, block, ("stat",))


def _with_behavioral(
    base: pd.DataFrame, week: int, prefixes: tuple[str, ...] = ("gap", "time", "cohort")
) -> pd.DataFrame:
    """Add inactivity, submission-timing and cohort-relative features."""
    path = get_data_path(f"processed/behavioral_week{week}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/build_extra_features.py --weeks {week}"
        )
    return _merge_block(base, pd.read_csv(path), prefixes)


def _with_assessment_weeks(
    base: pd.DataFrame, events: pd.DataFrame, week: int
) -> pd.DataFrame:
    """Restore per-week assessment resolution to a graph representation.

    The PCG state compresses every assessment into a handful of trajectory
    scalars (``week_mastery_mean``, ``frontier``, ``smoothness``, ...). That is
    lossless while a learner has assessment evidence in at most one week, but
    OULAD learners first hold a genuine *multi-week* assessment trajectory around
    week 8 -- which is precisely where the graph representation starts losing to
    the per-week baseline. This adds the per-week assessment columns back so the
    comparison isolates graph structure rather than assessment resolution.
    """
    columns = ["assess_attempts", "assess_score_mean", "assess_score_max", "assess_score_weighted"]
    work = events[KEY + ["week_index"]].copy()
    for column in columns:
        work[column] = _numeric(
            events[column] if column in events else None, events.index
        ).clip(lower=0)

    pivots = []
    for column in columns:
        pivot = work.pivot_table(
            index=KEY, columns="week_index", values=column, aggfunc="mean", fill_value=0.0
        )
        pivot = pivot.reindex(columns=range(1, week + 1), fill_value=0.0)
        pivot.columns = [f"{column}_week{w}" for w in pivot.columns]
        pivots.append(pivot)

    wide = pd.concat(pivots, axis=1).reset_index()
    merged = base.merge(wide, on=KEY, how="left")
    added = [c for c in wide.columns if c not in KEY]
    merged[added] = merged[added].fillna(0.0)
    return merged


def _with_traversal(base: pd.DataFrame, week: int, columns: tuple[str, ...]) -> pd.DataFrame:
    """Left-join cached per-learner traversal features onto a representation.

    The join is left so the base learner population is preserved exactly; the
    harness requires every representation in a run to cover the same learners,
    and traversal is derived from VLE events alone (a learner with only
    assessment evidence has no path). Absent paths become zeros, which is the
    correct reading: no observed traversal, not missing data.
    """
    path = get_data_path(f"processed/traversal_week{week}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/build_traversal_features.py --weeks {week}"
        )
    traversal = pd.read_csv(path)
    keep = [c for c in columns if c in traversal.columns]
    merged = base.merge(traversal[KEY + keep], on=KEY, how="left")
    merged[keep] = merged[keep].fillna(0.0)
    return merged


def prepare_dataset(dataset: str) -> None:
    """Build the minimal processed evidence tables required by this evaluator."""
    os.environ["PCG_DATASET"] = dataset
    if dataset == "kdd":
        from .kdd_preprocess import write_kdd_processed
        write_kdd_processed()
        return
    from .build_competency_graph import write_week_competencies_from_raw, write_prereq_edges_from_competencies
    from .evidence_mapping import write_assess_weekly_evidence_from_raw, write_vle_weekly_evidence_from_raw
    from .make_labels import make_labels
    make_labels()
    write_week_competencies_from_raw()
    write_prereq_edges_from_competencies()
    write_vle_weekly_evidence_from_raw()
    write_assess_weekly_evidence_from_raw()


def _hgb(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=random_state,
    )


def _scaled(estimator) -> Pipeline:
    """Standardise before distance- or gradient-based learners.

    Trees are scale-invariant and skip this; logistic regression, SVM, k-NN and
    the MLP are not, and comparing them unscaled would handicap them for reasons
    that have nothing to do with the method.
    """
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


#: Estimator per baseline family. These are faithful implementations of the
#: *approach families* common in the OULAD early-prediction literature -- profile
#: -based, clickstream-based, classical ML and a shallow neural net -- not
#: reproductions of specific published architectures. Every family is fitted on
#: the same folds and the same snapshot truncation as the proposed models, which
#: is the property that makes the comparison meaningful.
_ESTIMATORS: dict[str, "Callable[[int], object]"] = {
    "base_lr_profile": lambda s: _scaled(LogisticRegression(max_iter=2000, C=1.0)),
    "base_lr_clickstream": lambda s: _scaled(LogisticRegression(max_iter=2000, C=1.0)),
    "base_lr_weekly": lambda s: _scaled(LogisticRegression(max_iter=2000, C=1.0)),
    "base_rf": lambda s: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=s
    ),
    "base_rf_gini": lambda s: RandomForestClassifier(
        n_estimators=300, criterion="gini", min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=s,
    ),
    "base_rf_entropy": lambda s: RandomForestClassifier(
        n_estimators=300, criterion="entropy", min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=s,
    ),
    "base_svm": lambda s: _scaled(
        CalibratedClassifierCV(LinearSVC(C=0.1, dual="auto", max_iter=5000), cv=3)
    ),
    "base_knn": lambda s: _scaled(KNeighborsClassifier(n_neighbors=50, n_jobs=-1)),
    "base_mlp": lambda s: _scaled(
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                      random_state=s)
    ),
    "base_dffnn": lambda s: _scaled(
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=s,
        )
    ),
}


def _model(model_name: str, random_state: int):
    """Return the estimator for ``model_name``; gradient boosting by default."""
    factory = _ESTIMATORS.get(model_name)
    return factory(random_state) if factory is not None else _hgb(random_state)


def run_dataset_evaluation(
    dataset: str,
    snapshot_weeks: list[int],
    n_splits: int = 5,
    repeats: int = 5,
    random_state: int = 42,
    output_root: Path | None = None,
    verbose: bool = False,
    save_predictions: bool = False,
    models: tuple[str, ...] | None = None,
    event_transform: "Callable[[pd.DataFrame, int], pd.DataFrame] | None" = None,
    extra_columns: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated presentation-level CV and return fold metrics/predictions.

    ``event_transform`` is applied to the loaded weekly events before any
    representation is built, so every model in a run sees exactly the same
    (possibly degraded) evidence and comparisons stay paired.  ``extra_columns``
    are stamped onto each fold row, which lets a caller sweep a condition and
    concatenate the results.
    """
    os.environ["PCG_DATASET"] = dataset
    labels = pd.read_csv(get_data_path("processed/labels.csv"))
    output_root = output_root or Path("results") / "full_evaluation"

    # Fail fast: the group count is governed by the label population and does not
    # grow with the snapshot week, so validate it once up front rather than after
    # an earlier dataset has already been written to disk.
    label_groups = labels["code_module"].astype(str) + "::" + labels["code_presentation"].astype(str)
    if label_groups.nunique() < n_splits:
        raise ValueError(f"{dataset} has only {label_groups.nunique()} presentation groups; cannot use {n_splits} folds")

    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for week in snapshot_weeks:
        events = _load_weekly_events(week)
        if event_transform is not None:
            events = event_transform(events, week)
        representations = _representations(events, week, seed=random_state, models=models)

        # Prepare every representation on an identical, KEY-sorted row order so a
        # single CV split applies to all models. This makes the paired AUC
        # comparison exact and guards against silent learner-population drift
        # between representations.
        prepared: dict[str, tuple[pd.DataFrame, list[str]]] = {}
        reference_keys: list[tuple] | None = None
        for model_name, features in representations.items():
            feature_cols = [column for column in features.columns if column not in KEY + ["snapshot_week"]]
            if not feature_cols:
                if verbose:
                    print(f"[full-eval] dataset={dataset} week={week} model={model_name} skipped (no features)")
                continue
            frame = features.merge(labels, on=KEY, how="inner").sort_values(KEY).reset_index(drop=True)
            keys = list(map(tuple, frame[KEY].to_numpy()))
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise ValueError(
                    f"{dataset} week={week}: model {model_name} has a different learner population than its peers"
                )
            prepared[model_name] = (frame, feature_cols)

        if not prepared:
            continue

        # Labels and groups are identical across representations after the
        # alignment above, so derive the split inputs from any one of them.
        canonical_frame = prepared[next(iter(prepared))][0]
        y = canonical_frame["label"].astype(int).to_numpy()
        groups = canonical_frame["code_module"].astype(str) + "::" + canonical_frame["code_presentation"].astype(str)
        split_design = np.zeros((len(y), 1))

        week_auc: dict[str, list[float]] = {name: [] for name in prepared}
        for repeat in range(repeats):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
            for fold, (train_index, test_index) in enumerate(splitter.split(split_design, y, groups), start=1):
                if np.unique(y[test_index]).size < 2 or np.unique(y[train_index]).size < 2:
                    continue
                for model_name, (frame, feature_cols) in prepared.items():
                    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    estimator = _model(model_name, random_state + repeat * 100 + fold)
                    estimator.fit(x.iloc[train_index], y[train_index])
                    probability = estimator.predict_proba(x.iloc[test_index])[:, 1]
                    values = _metrics(y[test_index], probability)
                    week_auc[model_name].append(values["auc"])
                    fold_rows.append({
                        "dataset": dataset, "week": week, "model": model_name,
                        "repeat": repeat + 1, "fold": fold, "n_train": len(train_index),
                        "n_test": len(test_index), "n_features": len(feature_cols),
                        **(extra_columns or {}), **values,
                    })
                    if save_predictions:
                        prediction = frame.iloc[test_index][KEY].copy()
                        prediction["dataset"] = dataset
                        prediction["snapshot_week"] = week
                        prediction["model"] = model_name
                        prediction["repeat"] = repeat + 1
                        prediction["fold"] = fold
                        prediction["y_true"] = y[test_index]
                        prediction["proba_success"] = probability
                        pred_rows.append(prediction)
        if verbose:
            for model_name, aucs in week_auc.items():
                if aucs:
                    print(f"[full-eval] dataset={dataset} week={week} model={model_name} auc={np.mean(aucs):.4f}")
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    if save_predictions:
        predictions.to_csv(output_dir / "fold_predictions.csv", index=False)
    return folds, predictions


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a percentile bootstrap 95% CI *of the mean*.

    This replaces taking raw 2.5/97.5 percentiles of the fold metrics, which
    describe the spread of individual folds rather than the uncertainty of their
    mean. Note the folds within a repeat are not fully independent, so this is a
    mild under-estimate of the true interval; it is nonetheless an honest CI of
    the reported mean.
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return mean, float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def summarize_evaluation(folds: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    metrics = ["auc", "pr_auc_risk", "balanced_accuracy", "at_risk_recall", "brier", "ece"]
    rows = []
    for keys, group in folds.groupby(["dataset", "week", "model"]):
        record = dict(zip(["dataset", "week", "model"], keys))
        record["n_evaluations"] = len(group)
        for metric in metrics:
            mean, ci_low, ci_high = _bootstrap_mean_ci(group[metric].to_numpy())
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = ci_low
            record[f"{metric}_ci_high"] = ci_high
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values(["dataset", "week", "model"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary_metrics.csv", index=False)
    return summary


def summarize_paired_auc_differences(folds: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Summarise paired PCG-UT AUC gains over matched folds and repeats."""
    reference = folds[folds["model"] == "pcg_ut"]
    rows = []
    join_cols = ["dataset", "week", "repeat", "fold"]
    for model in ("flat_hgb", "temporal_hgb", "pcg_ut_no_graph", "pcg_ut_shuffled_graph", "pcg_ut_no_uncertainty"):
        other = folds[folds["model"] == model]
        paired = reference.merge(other, on=join_cols, suffixes=("_pcg_ut", "_baseline"))
        for (dataset, week), group in paired.groupby(["dataset", "week"]):
            deltas = (group["auc_pcg_ut"] - group["auc_baseline"]).to_numpy()
            delta_mean, delta_low, delta_high = _bootstrap_mean_ci(deltas)
            rows.append({
                "dataset": dataset, "week": week, "comparison": f"pcg_ut - {model}",
                "n_paired_folds": len(deltas), "auc_delta_mean": delta_mean,
                "auc_delta_ci_low": delta_low,
                "auc_delta_ci_high": delta_high,
            })
    result = pd.DataFrame(rows).sort_values(["dataset", "week", "comparison"])
    result.to_csv(output_root / "paired_auc_differences.csv", index=False)
    return result


def plot_evaluation(summary: pd.DataFrame, output_root: Path) -> list[Path]:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colours = {"flat_hgb": "#777777", "temporal_hgb": "#56b4e9", "pcg_ut_no_graph": "#e69f00",
               "pcg_ut": "#009e73", "pcg_ut_shuffled_graph": "#cc79a7", "pcg_ut_no_uncertainty": "#0072b2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dataset in zip(axes, ("oulab", "kdd")):
        subset = summary[(summary["dataset"] == dataset) & (summary["model"].isin(PRIMARY_MODELS))]
        for model, group in subset.groupby("model"):
            group = group.sort_values("week")
            ax.plot(group["week"], group["auc_mean"], marker="o", linewidth=2.5 if model == "pcg_ut" else 1.5,
                    label=model.replace("_", " "), color=colours[model])
            ax.fill_between(group["week"], group["auc_ci_low"], group["auc_ci_high"], color=colours[model], alpha=0.12)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Snapshot week")
        ax.set_ylabel("Group-CV AUC")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("PCG-UT: fair matched-head comparison", y=1.02)
    fig.tight_layout()
    main = figures / "auc_by_week.png"
    fig.savefig(main, dpi=220, bbox_inches="tight")
    plt.close(fig)

    final_week = summary.groupby("dataset")["week"].max().to_dict()
    ablation = summary[summary.apply(lambda r: r["week"] == final_week[r["dataset"]], axis=1)]
    ablation = ablation[ablation["model"].isin(("pcg_ut", "pcg_ut_no_graph", "pcg_ut_shuffled_graph", "pcg_ut_no_uncertainty"))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ("oulab", "kdd")):
        subset = ablation[ablation["dataset"] == dataset].sort_values("auc_mean")
        ax.barh([name.replace("pcg_ut", "PCG-UT").replace("_", " ") for name in subset["model"]], subset["auc_mean"],
                xerr=[subset["auc_mean"] - subset["auc_ci_low"], subset["auc_ci_high"] - subset["auc_mean"]],
                color=[colours[name] for name in subset["model"]], capsize=3)
        ax.set_title(f"{dataset.upper()} - final valid week")
        ax.set_xlabel("Group-CV AUC")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    ablation_path = figures / "final_week_ablations.png"
    fig.savefig(ablation_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [main, ablation_path]


def run_full_evaluation(
    datasets: list[str], n_splits: int = 5, repeats: int = 5, random_state: int = 42,
    output_root: Path | None = None, verbose: bool = False, prepare: bool = False,
    save_predictions: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    output_root = output_root or Path("results") / "full_evaluation"
    all_folds = []
    for dataset in datasets:
        if prepare:
            prepare_dataset(dataset)
        folds, _ = run_dataset_evaluation(dataset, DEFAULT_SNAPSHOTS[dataset], n_splits, repeats, random_state, output_root, verbose, save_predictions)
        all_folds.append(folds)
    combined = pd.concat(all_folds, ignore_index=True)
    combined.to_csv(output_root / "all_fold_metrics.csv", index=False)
    summary = summarize_evaluation(combined, output_root)
    summarize_paired_auc_differences(combined, output_root)
    figures = plot_evaluation(summary, output_root)
    protocol = {
        "datasets": datasets, "snapshot_weeks": {dataset: DEFAULT_SNAPSHOTS[dataset] for dataset in datasets},
        "n_splits": n_splits, "repeats": repeats, "random_state": random_state,
        "models": list(ALL_MODELS), "grouping": ["code_module", "code_presentation"],
        "saved_fold_predictions": save_predictions,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return summary, figures
