"""Week-0 learner features: everything known before any behaviour is observed.

The evaluation harness currently predicts week-2 outcomes from behavioural
evidence alone, ignoring the whole enrolment record.  That is the main reason
early snapshots are weak (AUC ~0.67 at week 2): registration timing, prior
attempts, course load and demographics are all available *before the course
starts*, and carry signal precisely where behavioural evidence is thinnest.

Leakage discipline
------------------
``studentRegistration.date_unregistration`` is excluded outright: it is populated
only for learners who withdrew, so it encodes the label directly (69% null,
exactly the non-withdrawn share).  ``studentInfo.final_result`` is the label
source and is likewise never read as a feature.  Everything below is fixed at
enrolment and observable at week 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import get_data_path

KEY = ["id_student", "code_module", "code_presentation"]

#: Ordered categories, so a single ordinal column preserves the ordering a
#: one-hot encoding would discard. Trees split ordinals efficiently.
_EDUCATION = [
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]
_IMD = [
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
_AGE = ["0-35", "35-55", "55<="]

#: Columns that must never be read as features.
FORBIDDEN = ("final_result", "date_unregistration")

#: Protected and protected-adjacent attributes.
#:
#: An early-warning system that flags learners partly on deprivation index or
#: disability status, and then routes finite instructor attention accordingly,
#: risks encoding structural inequality into the intervention itself. Isolating
#: these columns lets the evaluation report what accuracy actually costs.
#:
#: ``stat_education`` is included as protected-adjacent: prior attainment is a
#: legitimate academic predictor, but it is also a strong socioeconomic proxy, so
#: it is reported both ways rather than silently assigned to one side.
#: Region is included because geography proxies deprivation in the UK.
SENSITIVE_CORE = ("stat_imd", "stat_disability", "stat_female", "stat_age")
SENSITIVE_ADJACENT = ("stat_education",)


def sensitive_columns(frame: pd.DataFrame, include_adjacent: bool = True) -> list[str]:
    """Return the protected columns present in ``frame``."""
    names = set(SENSITIVE_CORE)
    if include_adjacent:
        names |= set(SENSITIVE_ADJACENT)
    return [
        c for c in frame.columns if c in names or c.startswith("stat_region_")
    ]


def _ordinal(series: pd.Series, categories: list[str]) -> pd.Series:
    """Map an ordered category to its rank, leaving unknown values missing."""
    lookup = {value: rank for rank, value in enumerate(categories)}
    return series.map(lookup).astype(float)


def build_static_features() -> pd.DataFrame:
    """Return one row per learner-presentation of week-0 observable features."""
    info = pd.read_csv(get_data_path("raw/studentInfo.csv"))
    registration = pd.read_csv(get_data_path("raw/studentRegistration.csv"))

    for column in FORBIDDEN:
        info = info.drop(columns=column, errors="ignore")
        registration = registration.drop(columns=column, errors="ignore")

    frame = info.merge(registration, on=KEY, how="left")
    out = frame[KEY].copy()

    # Registration timing. Negative = registered before the course opened;
    # positive = enrolled late, which is a materially different situation.
    days = pd.to_numeric(frame["date_registration"], errors="coerce")
    out["stat_registration_day"] = days
    out["stat_registered_late"] = (days > 0).astype(float)
    out["stat_registration_lead"] = (-days).clip(lower=0)

    # Prior history and course load.
    out["stat_prev_attempts"] = pd.to_numeric(frame["num_of_prev_attempts"], errors="coerce")
    out["stat_is_repeat"] = (out["stat_prev_attempts"] > 0).astype(float)
    out["stat_studied_credits"] = pd.to_numeric(frame["studied_credits"], errors="coerce")

    # Demographics.
    out["stat_education"] = _ordinal(frame["highest_education"], _EDUCATION)
    out["stat_imd"] = _ordinal(frame["imd_band"], _IMD)
    out["stat_age"] = _ordinal(frame["age_band"], _AGE)
    out["stat_female"] = (frame["gender"] == "F").astype(float)
    out["stat_disability"] = (frame["disability"] == "Y").astype(float)

    # Region is nominal with 13 levels; one-hot rather than impose a false order.
    region = pd.get_dummies(frame["region"], prefix="stat_region", dtype=float)
    out = pd.concat([out, region], axis=1)

    # Cohort context: how unusual is this learner's load within their cohort?
    # This is fixed at enrolment and helps transfer across presentations.
    grouped = out.groupby(["code_module", "code_presentation"])["stat_studied_credits"]
    out["stat_credits_vs_cohort"] = out["stat_studied_credits"] - grouped.transform("mean")

    return out


def load_static_features() -> pd.DataFrame:
    """Build (and cache) the week-0 feature table."""
    path = get_data_path("processed/static_features.csv")
    if path.exists():
        return pd.read_csv(path)
    features = build_static_features()
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features


if __name__ == "__main__":
    frame = build_static_features()
    path = get_data_path("processed/static_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    columns = [c for c in frame.columns if c.startswith("stat_")]
    print(f"{len(frame):,} learner-presentations, {len(columns)} features -> {path}")
    print(frame[columns].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
