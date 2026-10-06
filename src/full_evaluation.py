"""Feature representations and estimators for the cohort-exchange benchmark.

All models use the same cohort definitions and grouped evaluation protocol.
"""

from __future__ import annotations

from collections.abc import Callable

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
from .paths import get_data_path
from .pcg_ut import KEY, _state_features
from .pcg_ut_graph import build_pcg_ut_graph_features
from .features_static import sensitive_columns
from .traversal import ORDER_COLUMNS


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


def _representations(
    events: pd.DataFrame, week: int, seed: int = 42, models: tuple[str, ...] | None = None
) -> dict[str, pd.DataFrame]:
    """Build only the requested representations from the reported configurations."""
    builders = {
        "flat_hgb": lambda: _flat_features(events, week),
        "temporal_hgb": lambda: _temporal_features(events, week),
        "pcg_ut": lambda: _state_features(events, week, propagation="chain"),
        "base_lr_profile": lambda: _with_static(
            events[KEY].drop_duplicates().reset_index(drop=True)
        ),
        "base_lr_clickstream": lambda: _flat_features(events, week),
        "base_lr_weekly": lambda: _temporal_features(events, week),
        "base_rf_gini": lambda: _reference_all_features(events, week),
        "base_rf_entropy": lambda: _reference_all_features(events, week),
        "base_svm": lambda: _flat_features(events, week),
        "base_knn": lambda: _flat_features(events, week),
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
    """Add enrolment features, optionally excluding specified sensitive columns."""
    path = get_data_path("processed/static_features.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run python3 scripts/prepare_oulad.py")
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
            f"{path} missing; run python3 scripts/prepare_oulad.py"
        )
    return _merge_block(base, pd.read_csv(path), prefixes)


def _with_assessment_weeks(
    base: pd.DataFrame, events: pd.DataFrame, week: int
) -> pd.DataFrame:
    """Add weekly assessment columns to the graph representation."""
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


def _hgb(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=random_state,
    )


def _scaled(estimator) -> Pipeline:
    """Standardize inputs to linear, distance-based, and neural-network estimators."""
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
