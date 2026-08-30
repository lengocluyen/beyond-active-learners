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
    d = int(day)
    if d < 0:
        d = 0
    return (d // 7) + 1

def build_vle_weekly_evidence(
    student_vle: pd.DataFrame, vle: pd.DataFrame, competencies: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Output columns:
      id_student, code_module, code_presentation, week_index,
      clicks_total, active_days, clicks_by_activity_type_*
    """
    sv = student_vle.copy()
    sv["week_index"] = sv["date"].apply(day_to_week)

    # Join to get activity_type if you want richer evidence
    v = vle[["id_site", "activity_type"]].copy()
    sv = sv.merge(v, on="id_site", how="left")

    # Total clicks + active days
    base = (sv.groupby(["id_student", "code_module", "code_presentation", "week_index"])
              .agg(clicks_total=("sum_click", "sum"),
                   active_days=("date", "nunique"))
              .reset_index())

    # Optional: clicks by activity type (wide)
    piv = (sv.pivot_table(index=["id_student", "code_module", "code_presentation", "week_index"],
                          columns="activity_type",
                          values="sum_click",
                          aggfunc="sum",
                          fill_value=0)
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
    sa = student_assessment.copy()
    a = assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]].copy()
    df = sa.merge(a, on=["id_assessment"], how="left")

    # Prefer assessment date; if missing, fall back to submission date.
    df["assess_day"] = df["date"]
    if "date_submitted" in df.columns:
        df["assess_day"] = df["assess_day"].fillna(df["date_submitted"])
    df["week_index"] = df["assess_day"].apply(day_to_week)

    # Score normalization: score is typically 0..100; keep as 0..1 for fusion
    df["score_01"] = df["score"] / 100.0
    df["weight_01"] = df["weight"] / 100.0

    agg = (df.groupby(["id_student", "code_module", "code_presentation", "week_index"])
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
        wsum = (tmp.groupby(["id_student", "code_module", "code_presentation", "week_index"])
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
