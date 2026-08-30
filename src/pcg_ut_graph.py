"""PCG-UT-G: two-level Personal Competency Graph over week x activity-type nodes.

The original :mod:`src.pcg_ut` instantiates the PCG as a directed *path* over
progression units.  A path carries no structure beyond ordering, which the
``temporal_hgb`` baseline already encodes, so its graph ablations cannot
separate by construction.

This module instantiates a genuinely branching graph using evidence the dataset
already provides:

* ``WeekCompetency`` nodes ``W_w`` (one per progression unit), and
* ``ActivityCompetency`` nodes ``(w, a)`` for each VLE activity type ``a``,
  linked to their week by ``hasSubCompetency``.

Two propagation channels run over that DAG:

``temporal``
    ``(w, a) -> (w+1, a)`` -- the same modality across consecutive weeks.
``hierarchy``
    ``(w, ...) <-> W_w`` -- children inform the week node, the week node gently
    regularises its children.  This is the channel a path graph does not have.

Because activity types number ~20 rather than ~2, the shuffle ablation here is
non-degenerate; :func:`_permute` additionally refuses to return the identity.

Nodes remain dataset-grounded rather than domain competencies: activity types
are *modalities*, not skills.  This module therefore tests whether the PCG-UT
machinery can exploit branching structure at all, not whether OULAD supports a
domain competency ontology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import get_data_path

KEY = ["id_student", "code_module", "code_presentation"]

PROPAGATIONS = (
    "full",
    "none",
    "temporal_only",
    "hierarchy_only",
    "shuffled",
    # Non-destructive modes. These leave local node states untouched and vary
    # only which features are emitted, giving a strictly nested ladder:
    #     local_only  subset  none  subset  residual
    # Because each step only *adds* columns, a drop in AUC along the ladder can
    # only mean noise or overfitting -- never lost information. That separates
    # "does graph structure carry signal" from "does averaging destroy signal",
    # which the destructive modes above confound.
    "local_only",
    "residual",
    "residual_shuffled",
)

#: Modes that must not modify node states before features are read off.
_NON_DESTRUCTIVE = ("none", "local_only", "residual", "residual_shuffled")


def _activity_columns(frame: pd.DataFrame) -> list[str]:
    """Return the wide per-activity-type click columns, excluding the total."""
    return sorted(
        c for c in frame.columns if c.startswith("clicks_") and c != "clicks_total"
    )


def _permute(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a derangement-ish permutation that is never the identity.

    The original shuffle ablation drew ``permutation(t-1)`` with ``t`` as small
    as 2, which returns the identity outright and silently turned the ablation
    into a no-op.  Guarding here keeps the control meaningful.
    """
    if size < 2:
        return np.arange(size)
    identity = np.arange(size)
    for _ in range(32):
        candidate = rng.permutation(size)
        if not np.array_equal(candidate, identity):
            return candidate
    return np.roll(identity, 1)


def _load_weekly_events(snapshot_week: int) -> pd.DataFrame:
    vle = pd.read_csv(get_data_path("processed/vle_weekly_evidence.csv"))
    assess = pd.read_csv(get_data_path("processed/assess_weekly_evidence.csv"))
    events = vle.merge(
        assess, on=KEY + ["week_index"], how="outer", suffixes=("", "_assessment")
    )
    events["week_index"] = pd.to_numeric(events["week_index"], errors="coerce")
    events = events[events["week_index"].between(1, snapshot_week)].copy()
    events["week_index"] = events["week_index"].astype(int)
    return events


def _tensors(
    events: pd.DataFrame, snapshot_week: int, activity_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fold the long event table into ``(learner, week, activity)`` tensors."""
    numeric = activity_cols + ["active_days", "assess_attempts", "assess_score_mean"]
    for col in numeric:
        if col not in events.columns:
            events[col] = 0.0
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0.0)

    how = {c: "sum" for c in activity_cols}
    how.update(active_days="max", assess_attempts="sum", assess_score_mean="mean")
    agg = events.groupby(KEY + ["week_index"], as_index=False).agg(how)

    learners = agg[KEY].drop_duplicates().reset_index(drop=True)
    learners["_row"] = np.arange(len(learners))
    agg = agg.merge(learners, on=KEY, how="left")

    n, t, m = len(learners), snapshot_week, len(activity_cols)
    row = agg["_row"].to_numpy()
    week = agg["week_index"].to_numpy(dtype=int) - 1

    clicks = np.zeros((n, t, m), dtype=np.float64)
    clicks[row, week, :] = agg[activity_cols].to_numpy(dtype=np.float64)

    active_days = np.zeros((n, t), dtype=np.float64)
    active_days[row, week] = agg["active_days"].to_numpy(dtype=np.float64)

    attempts = np.zeros((n, t), dtype=np.float64)
    attempts[row, week] = agg["assess_attempts"].to_numpy(dtype=np.float64)

    score = np.full((n, t), 0.5, dtype=np.float64)
    score[row, week] = np.clip(agg["assess_score_mean"].to_numpy(dtype=np.float64), 0, 1)

    return learners.drop(columns="_row"), {
        "clicks": clicks,
        "active_days": active_days,
        "attempts": attempts,
        "score": score,
    }


def _local_states(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fuse raw evidence into per-node mastery and confidence.

    Mirrors the evidence model of :mod:`src.pcg_ut` so that the two remain
    comparable: behaviour is weak evidence, assessment is strong, and absent
    evidence stays uncertain instead of becoming negative evidence.
    """
    clicks = raw["clicks"]

    engagement = 1.0 - np.exp(-np.log1p(clicks) / 3.0)
    conf = 0.55 * (1.0 - np.exp(-clicks / 12.0))

    # Assessment evidence is week-level; it lifts the week node, not a modality.
    assess_conf = 1.0 - np.exp(-raw["attempts"])
    return {
        "mastery": engagement,
        "confidence": conf,
        "week_assess_conf": assess_conf,
        "week_assess_score": raw["score"],
    }


def _propagate(
    mastery: np.ndarray,
    confidence: np.ndarray,
    week_mastery: np.ndarray,
    week_conf: np.ndarray,
    propagation: str,
    rng: np.random.Generator,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run confidence-gated propagation over the two-level DAG.

    Returns the updated ``(mastery, confidence, week_mastery)`` tensors.  All
    transfer is gated on the *source* node's confidence, so an unobserved
    predecessor cannot manufacture evidence for its successor.
    """
    if propagation not in PROPAGATIONS:
        raise ValueError(f"propagation must be one of {PROPAGATIONS}")

    mastery, confidence = mastery.copy(), confidence.copy()
    week_mastery, week_conf = week_mastery.copy(), week_conf.copy()
    if propagation in _NON_DESTRUCTIVE:
        return mastery, confidence, week_mastery

    _, t, m = mastery.shape
    do_temporal = propagation in ("full", "temporal_only", "shuffled")
    do_hierarchy = propagation in ("full", "hierarchy_only", "shuffled")

    # Under `shuffled` the week->week+1 chain is preserved but the identity of
    # which modality feeds which is randomised, destroying the graph's structure
    # while leaving degree distribution and marginals untouched.
    lanes = _permute(m, rng) if propagation == "shuffled" else np.arange(m)

    for _ in range(passes):
        if do_temporal and t > 1:
            prev_m = mastery[:, :-1, lanes]
            prev_c = confidence[:, :-1, lanes]
            gate = 0.25 * prev_c
            mastery[:, 1:, :] = (1.0 - gate) * mastery[:, 1:, :] + gate * prev_m
            confidence[:, 1:, :] = np.maximum(confidence[:, 1:, :], 0.7 * prev_c)

        if do_hierarchy:
            # Children -> parent: confidence-weighted mean over modalities.
            denom = confidence.sum(axis=2)
            pooled = np.divide(
                (mastery * confidence).sum(axis=2),
                denom,
                out=np.full_like(denom, 0.5),
                where=denom > 1e-9,
            )
            child_conf = confidence.max(axis=2)
            week_mastery = np.where(
                week_conf + child_conf > 1e-9,
                (week_conf * week_mastery + child_conf * pooled)
                / np.maximum(week_conf + child_conf, 1e-9),
                week_mastery,
            )
            week_conf = np.maximum(week_conf, 0.7 * child_conf)

            # Parent -> children: gentle regularisation toward the week estimate.
            gate = (0.15 * week_conf)[:, :, None]
            mastery = (1.0 - gate) * mastery + gate * week_mastery[:, :, None]

    return mastery, confidence, week_mastery


def build_pcg_ut_graph_features(
    snapshot_week: int,
    propagation: str = "full",
    include_uncertainty: bool = True,
    seed: int = 42,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one two-level PCG state summary per learner at ``snapshot_week``."""
    if events is None:
        events = _load_weekly_events(snapshot_week)
    if events.empty:
        return pd.DataFrame(columns=KEY + ["snapshot_week"])

    activity_cols = _activity_columns(events)
    learners, raw = _tensors(events, snapshot_week, activity_cols)
    local = _local_states(raw)

    week_mastery = local["week_assess_score"]
    week_conf = local["week_assess_conf"]
    mastery, confidence, week_mastery = _propagate(
        local["mastery"],
        local["confidence"],
        week_mastery,
        week_conf,
        propagation=propagation,
        rng=np.random.default_rng(seed + snapshot_week),
    )

    t = snapshot_week
    observed = confidence > 0.05
    week_observed = week_conf > 0.05
    uncertainty = 1.0 - confidence

    out = learners.copy()
    out["snapshot_week"] = t

    # --- node-level summaries (comparable to PCG-UT) -----------------------
    out["pcgg_mastery_mean"] = mastery.mean(axis=(1, 2))
    out["pcgg_mastery_std"] = mastery.std(axis=(1, 2))
    out["pcgg_confidence_mean"] = confidence.mean(axis=(1, 2))
    out["pcgg_evidence_volume"] = np.log1p(raw["clicks"]).sum(axis=(1, 2))
    out["pcgg_recent_mastery"] = mastery[:, -1, :].mean(axis=1)
    out["pcgg_recent_confidence"] = confidence[:, -1, :].mean(axis=1)

    if include_uncertainty:
        out["pcgg_uncertainty_mean"] = uncertainty.mean(axis=(1, 2))
        out["pcgg_uncertainty_max"] = uncertainty.max(axis=(1, 2))

    # --- week-node (parent) summaries -------------------------------------
    out["pcgg_week_mastery_mean"] = week_mastery.mean(axis=1)
    out["pcgg_mastered_ratio"] = ((week_mastery >= 0.55) & week_observed).mean(axis=1)
    frontier = ((week_mastery >= 0.55) & week_observed) * np.arange(1, t + 1)[None, :]
    out["pcgg_frontier"] = frontier.max(axis=1)

    if t > 1:
        gap = np.maximum(0.0, week_mastery[:, 1:] - week_mastery[:, :-1])
        out["pcgg_prerequisite_gap"] = (gap * week_conf[:, 1:]).mean(axis=1)
        out["pcgg_smoothness"] = np.abs(np.diff(week_mastery, axis=1)).mean(axis=1)
    else:
        out["pcgg_prerequisite_gap"] = 0.0
        out["pcgg_smoothness"] = 0.0

    # --- structure-specific features a path graph cannot express ----------
    # Withheld under `local_only`, which is the true no-structure control: it
    # sees each node in isolation, with no cross-modality comparison at all.
    if propagation != "local_only":
        # Breadth: how much of the modality space carries any evidence.
        out["pcgg_breadth"] = observed.any(axis=1).mean(axis=1)
        # Dispersion of mastery across modalities, averaged over weeks.
        out["pcgg_modality_std"] = mastery.std(axis=2).mean(axis=1)
        # How unevenly effort is spread across modalities (normalised entropy).
        weight = raw["clicks"].sum(axis=1)
        share = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-9)
        entropy = -(share * np.log(np.maximum(share, 1e-12))).sum(axis=1)
        out["pcgg_modality_entropy"] = entropy / np.log(max(len(activity_cols), 2))
        # Disagreement between the week estimate and its children.
        out["pcgg_parent_child_gap"] = np.abs(
            week_mastery[:, :, None] - mastery
        ).mean(axis=(1, 2))

    # --- residual (non-destructive) graph information ---------------------
    if propagation in ("residual", "residual_shuffled"):
        lanes = (
            _permute(len(activity_cols), np.random.default_rng(seed + snapshot_week))
            if propagation == "residual_shuffled"
            else np.arange(len(activity_cols))
        )
        out = _residual_features(out, mastery, confidence, lanes, t)

    return out


def _residual_features(
    out: pd.DataFrame,
    mastery: np.ndarray,
    confidence: np.ndarray,
    lanes: np.ndarray,
    t: int,
) -> pd.DataFrame:
    """Append graph-derived *residuals* without altering the node states.

    Each residual answers "how far does this node sit from what its neighbours
    would predict".  Weighting by the neighbour's confidence keeps unobserved
    predecessors from manufacturing a deviation.  Nothing here overwrites
    ``mastery``, so the emitted representation is a strict superset of the
    non-residual one and the comparison isolates added information.
    """
    # Temporal residual: deviation from the learner's own previous week in the
    # same modality. Negative mass is separated out because *regression* is a
    # qualitatively different signal from progress.
    if t > 1:
        previous_m = mastery[:, :-1, lanes]
        previous_c = confidence[:, :-1, lanes]
        delta = mastery[:, 1:, :] - previous_m
        denominator = np.maximum(previous_c.sum(axis=(1, 2)), 1e-9)
        out["pcgg_res_t_mean"] = (delta * previous_c).sum(axis=(1, 2)) / denominator
        out["pcgg_res_t_absmean"] = (np.abs(delta) * previous_c).sum(axis=(1, 2)) / denominator
        out["pcgg_res_t_negative"] = (
            np.minimum(delta, 0.0) * previous_c
        ).sum(axis=(1, 2)) / denominator
    else:
        out["pcgg_res_t_mean"] = 0.0
        out["pcgg_res_t_absmean"] = 0.0
        out["pcgg_res_t_negative"] = 0.0

    # Hierarchical residual: how far each modality sits from the learner's own
    # week-level estimate. This is the two-level information a path cannot hold.
    weights = confidence.sum(axis=2, keepdims=True)
    pooled = np.divide(
        (mastery * confidence).sum(axis=2, keepdims=True),
        weights,
        out=np.full_like(weights, 0.5),
        where=weights > 1e-9,
    )
    hierarchical = mastery - pooled
    out["pcgg_res_h_absmean"] = np.abs(hierarchical).mean(axis=(1, 2))
    out["pcgg_res_h_std"] = hierarchical.std(axis=(1, 2))
    out["pcgg_res_h_max"] = hierarchical.max(axis=(1, 2))
    out["pcgg_res_h_min"] = hierarchical.min(axis=(1, 2))
    return out
