"""PCG-UT: uncertainty-aware temporal Personal Competency Graph baseline.

This module deliberately uses only numpy, pandas and scikit-learn so that it
can be run from the project's declared requirements.  It is a reproducible
first implementation of the PCG-UT architecture:

* weekly behavioural and assessment events are fused into learner-node states;
* each state stores a mastery proxy, confidence and uncertainty;
* prerequisite information is propagated along the course progression graph;
* a calibrated risk head is trained only on graph-state summaries.

The current public datasets do not expose item-to-skill mappings. Consequently
their nodes remain course progression units.  The implementation keeps the
state-construction API separate so that genuine competency mappings can replace
weekly nodes without changing the experiment runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .paths import get_data_path
from .splitters import group_train_test_split


KEY = ["id_student", "code_module", "code_presentation"]


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a finite numeric column, or a zero vector when it is unavailable."""
    if name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_weekly_events(snapshot_week: int) -> pd.DataFrame:
    vle = pd.read_csv(get_data_path("processed/vle_weekly_evidence.csv"))
    assess = pd.read_csv(get_data_path("processed/assess_weekly_evidence.csv"))
    join_key = KEY + ["week_index"]
    events = vle.merge(assess, on=join_key, how="outer", suffixes=("", "_assessment"))
    events["week_index"] = pd.to_numeric(events["week_index"], errors="coerce")
    events = events[events["week_index"].between(1, snapshot_week)].copy()
    events["week_index"] = events["week_index"].astype(int)
    return events


def _state_features(
    events: pd.DataFrame,
    snapshot_week: int,
    propagation: str = "chain",
    include_uncertainty: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build one PCG-UT representation per active learner at a checkpoint.

    The graph is a directed prerequisite chain over the available progression
    units.  Evidence is fused locally before two confidence-weighted forward
    propagation passes.  Missing evidence stays uncertain rather than becoming
    negative evidence.
    """
    if events.empty:
        return pd.DataFrame(columns=KEY + ["snapshot_week"])

    work = events[KEY + ["week_index"]].copy()
    clicks = _safe_col(events, "clicks_total").clip(lower=0)
    active_days = _safe_col(events, "active_days").clip(lower=0)
    attempts = _safe_col(events, "assess_attempts").clip(lower=0)
    score = _safe_col(events, "assess_score_mean").clip(0, 1)

    # Behaviour is weak evidence: it informs engagement but cannot by itself
    # yield a fully certain mastery estimate. Assessment evidence is stronger.
    engagement = 1.0 - np.exp(-(np.log1p(clicks) + 0.35 * active_days) / 3.0)
    engagement_conf = 0.55 * (1.0 - np.exp(-(clicks / 12.0 + active_days / 3.0)))
    assessment_conf = 1.0 - np.exp(-attempts)
    confidence = (engagement_conf + assessment_conf).clip(0, 1)
    mastery = np.where(
        confidence > 0,
        (engagement_conf * engagement + assessment_conf * score) / confidence,
        0.5,
    )

    work["mastery"] = mastery
    work["confidence"] = confidence
    work["uncertainty"] = 1.0 - confidence
    work["engagement"] = engagement
    work["assessment_signal"] = np.where(assessment_conf > 0, score, 0.5)
    work["evidence_volume"] = np.log1p(clicks) + attempts

    learners = work[KEY].drop_duplicates().reset_index(drop=True)
    learner_index = pd.MultiIndex.from_frame(learners)
    states = work.set_index(KEY + ["week_index"])[
        ["mastery", "confidence", "uncertainty", "engagement", "assessment_signal", "evidence_volume"]
    ].groupby(level=KEY + ["week_index"]).mean()

    n, t = len(learners), snapshot_week
    matrices: dict[str, np.ndarray] = {}
    for column, default in [("mastery", 0.5), ("confidence", 0.0), ("uncertainty", 1.0),
                            ("engagement", 0.0), ("assessment_signal", 0.5), ("evidence_volume", 0.0)]:
        pivot = states[column].unstack("week_index").reindex(index=learner_index, columns=range(1, t + 1))
        matrices[column] = pivot.fillna(default).to_numpy(dtype=np.float64)

    mastery_matrix = matrices["mastery"].copy()
    confidence_matrix = matrices["confidence"].copy()
    # Directed prerequisite propagation: a confident preceding competency can
    # gently regularise its successor.  It is deliberately conservative.
    if propagation not in {"chain", "none", "shuffled"}:
        raise ValueError("propagation must be one of: chain, none, shuffled")
    if propagation != "none" and t > 1:
        # Shuffling breaks the temporal prerequisite ordering while retaining the
        # same degree distribution, making it a meaningful graph ablation.
        predecessor = np.arange(t - 1)
        if propagation == "shuffled":
            predecessor = np.random.default_rng(seed + snapshot_week).permutation(t - 1)
        for _ in range(2):
            previous_m = mastery_matrix[:, predecessor].copy()
            previous_c = confidence_matrix[:, predecessor].copy()
            gate = 0.25 * previous_c
            mastery_matrix[:, 1:] = (1.0 - gate) * mastery_matrix[:, 1:] + gate * previous_m
            confidence_matrix[:, 1:] = np.maximum(confidence_matrix[:, 1:], 0.7 * previous_c)
    uncertainty_matrix = 1.0 - confidence_matrix

    observed = confidence_matrix > 0.05
    prerequisite_gap = np.maximum(0.0, mastery_matrix[:, 1:] - mastery_matrix[:, :-1])
    weighted_gap = prerequisite_gap * confidence_matrix[:, 1:]
    frontier = (observed & (mastery_matrix >= 0.55)) * np.arange(1, t + 1)[None, :]

    out = learners.copy()
    out["snapshot_week"] = snapshot_week
    out["pcgut_mastery_mean"] = mastery_matrix.mean(axis=1)
    out["pcgut_mastery_std"] = mastery_matrix.std(axis=1)
    out["pcgut_confidence_mean"] = confidence_matrix.mean(axis=1)
    out["pcgut_uncertainty_mean"] = uncertainty_matrix.mean(axis=1)
    out["pcgut_uncertainty_max"] = uncertainty_matrix.max(axis=1)
    out["pcgut_frontier"] = frontier.max(axis=1)
    out["pcgut_mastered_ratio"] = ((mastery_matrix >= 0.55) & observed).mean(axis=1)
    out["pcgut_prerequisite_gap"] = weighted_gap.mean(axis=1) if t > 1 else 0.0
    out["pcgut_smoothness"] = np.abs(np.diff(mastery_matrix, axis=1)).mean(axis=1) if t > 1 else 0.0
    out["pcgut_engagement_mean"] = matrices["engagement"].mean(axis=1)
    out["pcgut_assessment_mean"] = matrices["assessment_signal"].mean(axis=1)
    out["pcgut_evidence_volume"] = matrices["evidence_volume"].sum(axis=1)
    out["pcgut_recent_confidence"] = confidence_matrix[:, -1]
    out["pcgut_recent_mastery"] = mastery_matrix[:, -1]
    if not include_uncertainty:
        out = out.drop(columns=["pcgut_uncertainty_mean", "pcgut_uncertainty_max"])
    return out


def build_pcg_ut_features(
    snapshot_week: int,
    propagation: str = "chain",
    include_uncertainty: bool = True,
) -> pd.DataFrame:
    """Create PCG-UT features using only events observed by ``snapshot_week``."""
    return _state_features(
        _load_weekly_events(snapshot_week),
        snapshot_week,
        propagation=propagation,
        include_uncertainty=include_uncertainty,
    )


def _write_result(results_dir: Path, week: int, metrics: dict, predictions: pd.DataFrame, model: object) -> list[Path]:
    metrics_dir, preds_dir, models_dir = (results_dir / name for name in ("metrics", "predictions", "models"))
    for directory in (metrics_dir, preds_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"results_week{week}.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    existing["pcg_ut"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    preds_path = preds_dir / f"predictions_week{week}.csv"
    if preds_path.exists():
        previous = pd.read_csv(preds_path)
        if "model" in previous.columns:
            previous = previous[previous["model"] != "pcg_ut"]
        predictions = pd.concat([previous, predictions], ignore_index=True)
    predictions.to_csv(preds_path, index=False)
    model_path = models_dir / f"model_pcg_ut_week{week}.joblib"
    joblib.dump(model, model_path)
    return [metrics_path, preds_path, model_path]


def train_eval_pcg_ut_weekly(snapshot_weeks: list[int], random_state: int = 42, verbose: bool = False) -> list[Path]:
    """Train and evaluate PCG-UT using presentation-level held-out groups."""
    labels = pd.read_csv(get_data_path("processed/labels.csv"))
    results_dir = get_data_path("results")
    outputs: list[Path] = []
    for week in snapshot_weeks:
        features = build_pcg_ut_features(week)
        df = features.merge(labels, on=KEY, how="inner")
        train_df, test_df = group_train_test_split(df, ["code_module", "code_presentation"], "label")
        if train_df.empty or test_df.empty or train_df["label"].nunique() < 2:
            continue
        feature_cols = [col for col in features.columns if col not in KEY + ["snapshot_week"]]
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        )
        model.fit(train_df[feature_cols], train_df["label"].astype(int))
        proba = model.predict_proba(test_df[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_true = test_df["label"].astype(int).to_numpy()
        metrics = {
            "week": week,
            "model": "pcg_ut",
            "auc": float(roc_auc_score(y_true, proba)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "n_test": int(len(y_true)),
            "n_features": len(feature_cols),
            "description": "uncertainty-aware temporal PCG fusion with prerequisite-chain propagation",
        }
        if verbose:
            print(f"[PCG-UT] week={week} n_train={len(train_df)} n_test={len(test_df)} auc={metrics['auc']:.4f}")
        prediction_rows = test_df[KEY].copy()
        prediction_rows["snapshot_week"] = week
        prediction_rows["model"] = "pcg_ut"
        prediction_rows["y_true"] = y_true
        prediction_rows["proba"] = proba
        prediction_rows["pred"] = pred
        outputs.extend(_write_result(results_dir, week, metrics, prediction_rows, model))
    return outputs


if __name__ == "__main__":
    train_eval_pcg_ut_weekly(snapshot_weeks=[2, 4, 6, 8], verbose=True)
