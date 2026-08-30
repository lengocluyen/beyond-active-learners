"""Per-learner traversal structure over the resource graph.

Every graph instantiation tried so far gives each learner *identical* topology
and varies only the node values.  A fixed graph applied to per-learner values is
a fixed feature transform, so it contributes no between-learner structural
variance -- which is the mathematical reason the ablations in
``results/graph_ablation/`` cannot separate.

This module derives structure that genuinely differs per learner: the order in
which they actually traverse the material, compared against the order the cohort
follows.

Canonical order
---------------
``vle.csv`` populates ``week_from`` for only ~18% of resources, so the published
metadata cannot supply a domain ordering.  Instead each resource is ranked by the
cohort's median access day within its own presentation.  This is label-free but
*transductive*: it reads the interaction timestamps of every learner in the
presentation, including held-out ones.  No outcome labels are touched, and the
ordering is a property of the material rather than of any learner, but the
dependence should be disclosed in any write-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import get_data_path

KEY = ["id_student", "code_module", "code_presentation"]
PRESENTATION = ["code_module", "code_presentation"]

#: Order-only features. These describe the *shape* of a learner's path and are
#: invariant to how much evidence they generated, so a gain from this set cannot
#: be re-attributed to volume the flat baselines already see.
ORDER_COLUMNS = (
    "trav_reach",
    "trav_position_mean",
    "trav_position_std",
    "trav_backtrack_rate",
    "trav_below_frontier_rate",
    "trav_revisit_rate",
    "trav_step_absmean",
    "trav_forward_jump_mean",
    "trav_order_agreement",
)

#: Volume features, kept separate precisely because they overlap with what the
#: flat and temporal baselines already encode.
VOLUME_COLUMNS = ("trav_n_events", "trav_n_distinct", "trav_coverage")

_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}


def load_events(snapshot_week: int | None = None) -> pd.DataFrame:
    """Load resource-level VLE events, optionally truncated at a snapshot week."""
    events = pd.read_csv(get_data_path("raw/studentVle.csv"), dtype=_DTYPES)
    events = events[events["date"] >= 0]
    events["week_index"] = (events["date"] // 7).astype("int16") + 1
    if snapshot_week is not None:
        events = events[events["week_index"] <= snapshot_week]
    return events


def canonical_order(events: pd.DataFrame) -> pd.DataFrame:
    """Rank resources by the cohort's median access day within each presentation.

    Ranking is dense so that resources typically met together share a rank, which
    keeps the "did this learner move forward" question about genuine progression
    rather than about arbitrary tie-breaking.
    """
    median_day = (
        events.groupby(PRESENTATION + ["id_site"], observed=True)["date"]
        .median()
        .reset_index(name="canonical_day")
    )
    median_day["canonical_rank"] = (
        median_day.groupby(PRESENTATION, observed=True)["canonical_day"]
        .rank(method="dense")
        .astype(float)
    )
    # Normalise so ranks are comparable across presentations of different length.
    size = median_day.groupby(PRESENTATION, observed=True)["canonical_rank"].transform("max")
    median_day["canonical_norm"] = median_day["canonical_rank"] / size.clip(lower=1)
    return median_day


def traversal_features(snapshot_week: int, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise each learner's path through the material up to ``snapshot_week``.

    All features describe *order*, not volume, so they are not recoverable from
    the weekly aggregate counts the other representations consume.
    """
    if events is None:
        events = load_events(snapshot_week)
    else:
        events = events[events["week_index"] <= snapshot_week]

    order = canonical_order(events)
    events = events.merge(order, on=PRESENTATION + ["id_site"], how="left")
    events = events.dropna(subset=["canonical_norm"])

    # Learner's own sequence: by day, then by canonical position within a day
    # (intra-day ordering is unobserved, so this is the neutral tie-break).
    events = events.sort_values(KEY + ["date", "canonical_norm"], kind="stable")

    grouped = events.groupby(KEY, observed=True, sort=False)
    position = events["canonical_norm"].to_numpy()

    # Step-to-step movement along the canonical axis.
    step = grouped["canonical_norm"].diff().to_numpy()
    same_learner = ~np.isnan(step)
    events["_step"] = np.where(same_learner, step, 0.0)
    events["_is_step"] = same_learner.astype(float)
    events["_backtrack"] = ((events["_step"] < 0) & same_learner).astype(float)
    events["_forward_jump"] = np.where(events["_step"] > 0, events["_step"], 0.0)

    # Running maximum: an access below it is a return to earlier material.
    events["_running_max"] = grouped["canonical_norm"].cummax()
    events["_below_frontier"] = (position < events["_running_max"] - 1e-9).astype(float)

    # Revisits to a resource already seen by this learner.
    events["_seen"] = grouped["id_site"].transform(lambda s: s.duplicated()).astype(float)

    aggregated = events.groupby(KEY, observed=True).agg(
        trav_n_events=("canonical_norm", "size"),
        trav_n_distinct=("id_site", "nunique"),
        trav_reach=("canonical_norm", "max"),
        trav_position_mean=("canonical_norm", "mean"),
        trav_position_std=("canonical_norm", "std"),
        trav_backtrack_rate=("_backtrack", "mean"),
        trav_below_frontier_rate=("_below_frontier", "mean"),
        trav_revisit_rate=("_seen", "mean"),
        trav_step_absmean=("_step", lambda s: float(np.abs(s).mean())),
        trav_forward_jump_mean=("_forward_jump", "mean"),
    )

    # Rank correlation between the learner's own sequence and the cohort order.
    # A learner tracking the curriculum scores near 1; an erratic path near 0.
    def _order_agreement(series: pd.Series) -> float:
        if len(series) < 3:
            return np.nan
        own = np.arange(len(series), dtype=float)
        other = series.to_numpy(dtype=float)
        if np.std(other) < 1e-12:
            return np.nan
        return float(np.corrcoef(own, pd.Series(other).rank().to_numpy())[0, 1])

    agreement = grouped["canonical_norm"].apply(_order_agreement).rename("trav_order_agreement")
    aggregated = aggregated.join(agreement)

    # Coverage: share of the material the cohort had reached by this week that
    # the learner actually visited. Separates "behind" from "erratic".
    reached = (
        events.groupby(PRESENTATION, observed=True)["id_site"].nunique().rename("_available")
    )
    aggregated = aggregated.join(reached, on=PRESENTATION)
    aggregated["trav_coverage"] = aggregated["trav_n_distinct"] / aggregated["_available"].clip(lower=1)
    aggregated = aggregated.drop(columns="_available")

    aggregated["trav_position_std"] = aggregated["trav_position_std"].fillna(0.0)
    aggregated["trav_order_agreement"] = aggregated["trav_order_agreement"].fillna(0.0)

    out = aggregated.reset_index()
    # `_DTYPES` reads the module/presentation codes as `category` to save memory
    # over ~10M rows, but that dtype survives groupby/reset_index and then makes
    # downstream merges against object-dtype keys fragile. Normalise on the way
    # out; the memory win has already been banked during aggregation.
    for column in ("code_module", "code_presentation"):
        if column in out.columns:
            out[column] = out[column].astype(str)
    out["snapshot_week"] = snapshot_week
    return out
