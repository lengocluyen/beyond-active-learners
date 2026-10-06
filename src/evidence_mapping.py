from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from .paths import get_data_path

def day_to_week(day: int) -> int:
    # OULAD "date" is typically integer days (may be negative for pre-start access).
    # We map day 0..6 -> week 1, day 7..13 -> week 2, etc.
    if pd.isna(day):
        return np.nan
    if day < 0:
        return np.nan
    d = int(day)
    return (d // 7) + 1


def admissible_assessment_submissions(student_assessment: pd.DataFrame) -> pd.DataFrame:
    """Current-presentation submissions with a recorded, nonnegative event day.

    Banked results describe credit carried from a previous presentation, not a
    submission event in this presentation. Missing/invalid submission dates
    cannot be replaced by the scheduled deadline. Upper-cutoff filtering is
    applied by the weekly loader or by the caller's explicit snapshot window.
    """
    required = {"date_submitted", "is_banked"}
    missing = required.difference(student_assessment.columns)
    if missing:
        raise ValueError(f"Assessment evidence requires columns: {sorted(missing)}")
    frame = student_assessment.copy()
    frame["date_submitted"] = pd.to_numeric(frame["date_submitted"], errors="coerce")
    banked = pd.to_numeric(frame["is_banked"], errors="coerce")
    valid_day = np.isfinite(frame["date_submitted"]) & frame["date_submitted"].ge(0)
    return frame.loc[valid_day & banked.eq(0)].copy()

def build_vle_weekly_evidence(
    student_vle: pd.DataFrame, vle: pd.DataFrame, competencies: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Output columns:
      id_student, code_module, code_presentation, week_index,
      clicks_total, active_days, clicks_by_activity_type_*
    """
    sv = student_vle.copy()
    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv = sv.loc[np.isfinite(sv["date"]) & sv["date"].ge(0)].copy()
    sv["week_index"] = sv["date"].apply(day_to_week)

    # Join to get activity_type if you want richer evidence
    v = vle[["id_site", "activity_type"]].copy()
    sv = sv.merge(v, on="id_site", how="left", validate="many_to_one")

    # Total clicks + active days
    base = (sv.groupby(["id_student", "code_module", "code_presentation", "week_index"], observed=True)
              .agg(clicks_total=("sum_click", "sum"),
                   active_days=("date", "nunique"))
              .reset_index())

    # Optional: clicks by activity type (wide)
    piv = (sv.pivot_table(index=["id_student", "code_module", "code_presentation", "week_index"],
                          columns="activity_type",
                          values="sum_click",
                          aggfunc="sum",
                          fill_value=0, observed=True)
             .reset_index())

    # Prefix activity columns
    activity_cols = [c for c in piv.columns if c not in ["id_student", "code_module", "code_presentation", "week_index"]]
    piv = piv.rename(columns={c: f"clicks_{c}" for c in activity_cols})

    out = base.merge(piv, on=["id_student", "code_module", "code_presentation", "week_index"], how="left")

    if competencies is not None:
        comp_cols = ["competency_id", "code_module", "code_presentation", "week_index"]
        missing = [c for c in comp_cols if c not in competencies.columns]
        if missing:
            raise ValueError(f"competencies missing columns: {missing}")
        out = out.merge(
            competencies[comp_cols],
            on=["code_module", "code_presentation", "week_index"],
            how="left",
        )
    return out

def build_assess_weekly_evidence(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    competencies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Output columns:
      id_student, code_module, code_presentation, week_index,
      assess_attempts, assess_score_mean, assess_score_max, assess_score_weighted
    """
    sa = admissible_assessment_submissions(student_assessment)
    a = assessments[["id_assessment", "code_module", "code_presentation", "weight"]].copy()
    df = sa.merge(a, on="id_assessment", how="left", validate="many_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError("Submission has no matching assessment metadata")
    df = df.drop(columns="_merge")

    # Week w contains days [7*(w-1), 7*w). Consequently the downstream
    # week_index <= snapshot_week filter excludes submissions on the cutoff.
    # A due date is schedule information, never evidence that a submission or
    # its eventual score already existed. No deadline fallback is permitted.
    df["assess_day"] = df["date_submitted"]
    df["week_index"] = df["assess_day"].apply(day_to_week)

    # OULAD has no grade-release timestamp. Scores here assume availability at
    # submission; this explicit assumption is recorded in preprocessing/run
    # manifests and is not proof of actual grade availability at that time.
    df["score_01"] = df["score"] / 100.0
    df["weight_01"] = df["weight"] / 100.0

    agg = (df.groupby(["id_student", "code_module", "code_presentation", "week_index"], observed=True)
             .agg(assess_attempts=("id_assessment", "count"),
                  assess_score_mean=("score_01", "mean"),
                  assess_score_max=("score_01", "max"),
                  assess_score_weighted=("score_01", lambda s: float(np.mean(s)))  # placeholder
                  )
             .reset_index())

    # Weighted score if weights exist per assessment row (better):
    tmp = df.dropna(subset=["weight_01"])
    if len(tmp) > 0:
        tmp = tmp.copy()
        tmp["weighted_score"] = tmp["score_01"] * tmp["weight_01"]
        wsum = (tmp.groupby(["id_student", "code_module", "code_presentation", "week_index"], observed=True)
                  .agg(weight_sum=("weight_01", "sum"),
                       score_weighted_sum=("weighted_score", "sum"))
                  .reset_index())
        wsum["assess_score_weighted"] = np.where(
            wsum["weight_sum"] > 0,
            wsum["score_weighted_sum"] / wsum["weight_sum"],
            np.nan,
        )
        wagg = wsum.drop(columns=["weight_sum", "score_weighted_sum"])
        agg = agg.drop(columns=["assess_score_weighted"]).merge(
            wagg,
            on=["id_student", "code_module", "code_presentation", "week_index"],
            how="left",
        )

    if competencies is not None:
        comp_cols = ["competency_id", "code_module", "code_presentation", "week_index"]
        missing = [c for c in comp_cols if c not in competencies.columns]
        if missing:
            raise ValueError(f"competencies missing columns: {missing}")
        agg = agg.merge(
            competencies[comp_cols],
            on=["code_module", "code_presentation", "week_index"],
            how="left",
        )

    return agg


def write_vle_weekly_evidence_from_raw(out_path: Path | None = None) -> Path:
    """
    Map studentVle events to week competencies and write vle_weekly_evidence.csv.
    """

    student_vle = pd.read_csv(get_data_path("raw/studentVle.csv"))
    vle = pd.read_csv(get_data_path("raw/vle.csv"))
    competencies = pd.read_csv(get_data_path("processed/competencies.csv"))

    out = build_vle_weekly_evidence(student_vle, vle, competencies=competencies)

    if out_path is None:
        out_path = get_data_path("processed/vle_weekly_evidence.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def write_assess_weekly_evidence_from_raw(out_path: Path | None = None) -> Path:
    """
    Map assessment attempts to week competencies and write assess_weekly_evidence.csv.
    """

    student_assessment = pd.read_csv(get_data_path("raw/studentAssessment.csv"))
    assessments = pd.read_csv(get_data_path("raw/assessments.csv"))
    competencies = pd.read_csv(get_data_path("processed/competencies.csv"))

    out = build_assess_weekly_evidence(
        student_assessment, assessments, competencies=competencies
    )

    if out_path is None:
        out_path = get_data_path("processed/assess_weekly_evidence.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    write_vle_weekly_evidence_from_raw()
    # Optional: also build assessment weekly evidence if raw files exist.
    # This will write data/processed/assess_weekly_evidence.csv
    write_assess_weekly_evidence_from_raw()
