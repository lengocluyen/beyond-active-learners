"""Behavioural timing features: when a learner acts, not how much.

The weekly aggregates the harness consumes destroy timing structure.  Summing
clicks per week hides whether a learner worked steadily or vanished for ten
days; averaging scores hides whether they submitted early or scraped past the
deadline.  Disengagement announces itself in *gaps* and *lateness* well before
it shows up in volume, which is exactly the regime early prediction cares about.

Three blocks, all truncated at the snapshot week:

``gap``
    Inactivity structure over day-level VLE events.
``timing``
    Submission earliness/lateness against assessment deadlines, plus missed
    assessments -- a due assessment with no submission is strong evidence.
``cohort``
    Position relative to the learner's own presentation, which normalises away
    the between-module shift that hurts cross-presentation generalisation.

The cohort block is transductive in the same sense as ``src/traversal``: it
reads other enrolled learners' evidence up to week *k*.  Splits are by
presentation, so a test presentation's percentiles are computed entirely from
test learners and training presentations' from their own -- there is no
train/test contamination, and the statistic is computable during a live course.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import get_data_path
from .evidence_mapping import admissible_assessment_submissions

KEY = ["id_student", "code_module", "code_presentation"]
PRESENTATION = ["code_module", "code_presentation"]

_VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}


def load_vle_events() -> pd.DataFrame:
    """Day-level VLE events with non-negative dates."""
    events = pd.read_csv(get_data_path("raw/studentVle.csv"), dtype=_VLE_DTYPES)
    return events[events["date"] >= 0]


def _plain_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the join keys back to plain dtypes.

    ``_VLE_DTYPES`` reads the module/presentation codes as ``category`` because
    it saves a lot of memory over ~10M rows, but the category dtype survives
    ``groupby(...).reset_index()`` and then leaks into two different failures:
    a blanket ``fillna`` raises (0.0 is not a category), and merging a
    categorical key against an object key read from a different CSV is fragile.
    Normalising here keeps the memory win during aggregation without paying for
    it downstream.
    """
    out = frame.copy()
    for column in ("code_module", "code_presentation"):
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def gap_features(snapshot_week: int, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inactivity structure up to ``snapshot_week``."""
    if events is None:
        events = load_vle_events()
    horizon = snapshot_week * 7
    window = events[(events["date"] >= 0) & (events["date"] < horizon)]

    daily = (
        window.groupby(KEY + ["date"], observed=True)["sum_click"]
        .sum()
        .reset_index()
        .sort_values(KEY + ["date"], kind="stable")
    )

    grouped = daily.groupby(KEY, observed=True)
    daily["_gap"] = grouped["date"].diff()

    aggregated = grouped.agg(
        gap_active_days=("date", "nunique"),
        gap_first_day=("date", "min"),
        gap_last_day=("date", "max"),
        gap_max=("_gap", "max"),
        gap_mean=("_gap", "mean"),
        gap_std=("_gap", "std"),
    )

    # Silence since the learner was last seen, measured from the snapshot edge.
    aggregated["gap_days_since_last"] = horizon - aggregated["gap_last_day"]
    aggregated["gap_span"] = aggregated["gap_last_day"] - aggregated["gap_first_day"]
    # Density of engagement across the span actually used.
    aggregated["gap_density"] = aggregated["gap_active_days"] / aggregated["gap_span"].clip(lower=1)

    # Momentum: clicks in the final fortnight against the prior fortnight. A
    # ratio below 1 means the learner is decelerating.
    recent = window[window["date"] >= horizon - 14]
    earlier = window[(window["date"] >= horizon - 28) & (window["date"] < horizon - 14)]
    recent_clicks = recent.groupby(KEY, observed=True)["sum_click"].sum().rename("_recent")
    earlier_clicks = earlier.groupby(KEY, observed=True)["sum_click"].sum().rename("_earlier")
    aggregated = aggregated.join(recent_clicks).join(earlier_clicks)
    aggregated[["_recent", "_earlier"]] = aggregated[["_recent", "_earlier"]].fillna(0.0)
    aggregated["gap_momentum"] = (aggregated["_recent"] + 1.0) / (aggregated["_earlier"] + 1.0)
    aggregated["gap_recent_clicks"] = np.log1p(aggregated["_recent"])
    aggregated = aggregated.drop(columns=["_recent", "_earlier"])

    aggregated["gap_max"] = aggregated["gap_max"].fillna(aggregated["gap_days_since_last"])
    aggregated["gap_mean"] = aggregated["gap_mean"].fillna(aggregated["gap_days_since_last"])
    aggregated["gap_std"] = aggregated["gap_std"].fillna(0.0)
    return _plain_keys(aggregated.reset_index())


def timing_features(snapshot_week: int) -> pd.DataFrame:
    """Submission earliness/lateness and missed assessments up to the snapshot.

    Built over the *full enrolment roster*, not just learners who submitted
    something. A learner with no submissions has missed every assessment due so
    far, which is the strongest signal in this block; deriving the table only
    from ``studentAssessment`` would drop those learners entirely and a
    downstream zero-fill would then record them as having missed nothing --
    exactly inverting the signal for the highest-risk group.
    """
    submissions = pd.read_csv(get_data_path("raw/studentAssessment.csv"))
    assessments = pd.read_csv(get_data_path("raw/assessments.csv"))
    roster = pd.read_csv(get_data_path("raw/studentInfo.csv"))[KEY].drop_duplicates()
    horizon = snapshot_week * 7

    schedule = assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]]
    schedule = schedule.rename(columns={"date": "due_day"})
    schedule["due_day"] = pd.to_numeric(schedule["due_day"], errors="coerce")

    frame = admissible_assessment_submissions(submissions).merge(
        schedule, on="id_assessment", how="left", validate="many_to_one"
    )
    # Apply the same event/banking convention as the weekly assessment block.
    frame = frame.loc[frame["date_submitted"] < horizon].copy()
    frame = frame.sort_values(["date_submitted", "id_assessment"], kind="stable")
    frame["_lateness"] = frame["date_submitted"] - frame["due_day"]
    frame["_submitted_due"] = frame["due_day"].lt(horizon).astype(int)

    grouped = frame.groupby(KEY, observed=True)
    submitted = grouped.agg(
        time_submissions=("id_assessment", "count"),
        time_lateness_mean=("_lateness", "mean"),
        time_lateness_max=("_lateness", "max"),
        time_lateness_last=("_lateness", "last"),
        time_late_rate=("_lateness", lambda s: float((s > 0).mean())),
        _submitted_weight=("weight", "sum"),
        _submitted_due=("_submitted_due", "sum"),
    )

    # Left-join onto the full roster so non-submitters are present with zeros
    # rather than absent.
    out = roster.merge(submitted, on=KEY, how="left")
    for column in ("time_submissions", "_submitted_weight", "_submitted_due"):
        out[column] = out[column].fillna(0.0)
    # Lateness stays 0 where nothing was submitted: no observation, not "on time".
    for column in ("time_lateness_mean", "time_lateness_max", "time_lateness_last",
                   "time_late_rate"):
        out[column] = out[column].fillna(0.0)

    # Earliness is the useful direction for on-time learners; keep it explicit
    # rather than relying on the sign of a mean that late learners dominate.
    out["time_earliness_mean"] = -out["time_lateness_mean"]

    # Missed assessments: due before the snapshot, never submitted. Counted
    # against each learner's own presentation schedule, and weighted, since
    # skipping a 30%-weight assignment is not the same as skipping a 2% one.
    due = schedule[schedule["due_day"] < horizon]
    due_counts = (
        due.groupby(PRESENTATION, observed=True)
        .agg(_due=("id_assessment", "count"), _due_weight=("weight", "sum"))
        .reset_index()
    )
    out = out.merge(due_counts, on=PRESENTATION, how="left")
    out[["_due", "_due_weight"]] = out[["_due", "_due_weight"]].fillna(0.0)

    # An early submission for a future deadline cannot cancel a different
    # assessment that is already due and still unsubmitted.
    out["time_missed"] = (out["_due"] - out["_submitted_due"]).clip(lower=0)
    out["time_missed_rate"] = out["time_missed"] / out["_due"].clip(lower=1)
    out["time_weight_completed"] = out["_submitted_weight"] / out["_due_weight"].clip(lower=1)
    out = out.drop(columns=["_due", "_due_weight", "_submitted_weight", "_submitted_due"])

    return _plain_keys(out)


def cohort_features(snapshot_week: int, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank each learner against their own presentation on core evidence."""
    if events is None:
        events = load_vle_events()
    horizon = snapshot_week * 7
    window = events[(events["date"] >= 0) & (events["date"] < horizon)]

    base = window.groupby(KEY, observed=True).agg(
        _clicks=("sum_click", "sum"),
        _days=("date", "nunique"),
        _sites=("id_site", "nunique"),
    ).reset_index()

    for source, name in (("_clicks", "clicks"), ("_days", "days"), ("_sites", "sites")):
        grouped = base.groupby(PRESENTATION, observed=True)[source]
        base[f"cohort_{name}_pct"] = grouped.rank(pct=True)
        # Standardised as well as ranked: the two disagree under heavy skew,
        # and which one carries the signal is an empirical question.
        base[f"cohort_{name}_z"] = (base[source] - grouped.transform("mean")) / grouped.transform(
            "std"
        ).replace(0.0, np.nan)

    base = base.drop(columns=["_clicks", "_days", "_sites"])
    # Fill only the derived numeric columns. A blanket fillna would also target
    # the categorical join keys, which raises rather than being a no-op.
    derived = [c for c in base.columns if c.startswith("cohort_")]
    base[derived] = base[derived].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _plain_keys(base)


def build_behavioral_features(
    snapshot_week: int, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join the gap, timing and cohort blocks for one snapshot week."""
    if events is None:
        events = load_vle_events()
    frame = gap_features(snapshot_week, events)
    frame = frame.merge(timing_features(snapshot_week), on=KEY, how="outer")
    frame = frame.merge(cohort_features(snapshot_week, events), on=KEY, how="outer")
    frame["snapshot_week"] = snapshot_week
    return frame
