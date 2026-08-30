from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import get_data_path

def build_week_competencies(courses: pd.DataFrame, max_weeks_default: int = 30) -> pd.DataFrame:
    """
    Return competencies table:
    competency_id, code_module, code_presentation, week_index, name
    """
    rows = []
    for _, row in courses.iterrows():
        mod = row["code_module"]
        pres = row["code_presentation"]
        # If you have module_presentation_length, use it; else fallback
        n_weeks = int(row.get("module_presentation_length", max_weeks_default))
        n_weeks = max(1, n_weeks)
        for w in range(1, n_weeks + 1):
            cid = f"{mod}_{pres}_W{w:02d}"
            rows.append({
                "competency_id": cid,
                "code_module": mod,
                "code_presentation": pres,
                "week_index": w,
                "name": f"{mod}-{pres} week {w}"
            })
    return pd.DataFrame(rows)


def build_week_competencies_from_vle(
    vle: pd.DataFrame, student_vle: pd.DataFrame
) -> pd.DataFrame:
    """
    Build weekly competencies using VLE metadata and interaction dates.

    Derives the week range per (code_module, code_presentation) from:
    - VLE metadata: week_from / week_to
    - Student interactions: date (days since start) -> week_index = floor(date/7) + 1
    """

    key = ["code_module", "code_presentation"]
    for col in key + ["week_from", "week_to"]:
        if col not in vle.columns:
            raise ValueError(f"vle missing column: {col}")
    for col in key + ["date"]:
        if col not in student_vle.columns:
            raise ValueError(f"student_vle missing column: {col}")

    # VLE metadata weeks (ignore negative/zero weeks).
    vle_weeks = vle[key + ["week_from", "week_to"]].copy()
    vle_weeks["week_from"] = pd.to_numeric(vle_weeks["week_from"], errors="coerce")
    vle_weeks["week_to"] = pd.to_numeric(vle_weeks["week_to"], errors="coerce")
    vle_weeks = vle_weeks.dropna(subset=["week_from", "week_to"])
    vle_weeks = vle_weeks[(vle_weeks["week_from"] > 0) | (vle_weeks["week_to"] > 0)]

    vle_min = vle_weeks.groupby(key)["week_from"].min()
    vle_max = vle_weeks.groupby(key)["week_to"].max()

    # Interaction weeks from studentVle (days since start -> week index).
    sv = student_vle[key + ["date"]].copy()
    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv = sv.dropna(subset=["date"])
    sv = sv[sv["date"] >= 0]
    sv["week_index"] = (sv["date"] // 7).astype(int) + 1
    sv_min = sv.groupby(key)["week_index"].min()
    sv_max = sv.groupby(key)["week_index"].max()

    # Combine ranges from both sources.
    ranges = pd.concat(
        [
            vle_min.rename("min_week"),
            vle_max.rename("max_week"),
            sv_min.rename("min_week_sv"),
            sv_max.rename("max_week_sv"),
        ],
        axis=1,
    ).reset_index()

    ranges["min_week"] = ranges[["min_week", "min_week_sv"]].min(axis=1, skipna=True)
    ranges["max_week"] = ranges[["max_week", "max_week_sv"]].max(axis=1, skipna=True)

    rows = []
    for _, row in ranges.iterrows():
        mod = row["code_module"]
        pres = row["code_presentation"]
        min_w = int(row["min_week"]) if pd.notna(row["min_week"]) else 1
        max_w = int(row["max_week"]) if pd.notna(row["max_week"]) else min_w
        min_w = max(1, min_w)
        max_w = max(min_w, max_w)
        for w in range(min_w, max_w + 1):
            cid = f"{mod}_{pres}_W{w:02d}"
            rows.append(
                {
                    "competency_id": cid,
                    "code_module": mod,
                    "code_presentation": pres,
                    "week_index": w,
                    "name": f"{mod}-{pres} week {w}",
                }
            )

    return pd.DataFrame(rows)


def write_week_competencies_from_raw(out_path: Path | None = None) -> Path:
    """
    Read raw OULAD VLE files and write competencies.csv.
    """

    vle = pd.read_csv(get_data_path("raw/vle.csv"))
    student_vle = pd.read_csv(get_data_path("raw/studentVle.csv"))
    competencies = build_week_competencies_from_vle(vle, student_vle)

    if out_path is None:
        out_path = get_data_path("processed/competencies.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    competencies.to_csv(out_path, index=False)
    return out_path


def build_prereq_edges(competencies: pd.DataFrame) -> pd.DataFrame:
    """
    Return edges table:
    source_competency_id, relation, target_competency_id
    where relation = prerequisiteOf (week w -> week w+1)
    """
    edges = []
    grp_cols = ["code_module", "code_presentation"]
    for (mod, pres), g in competencies.groupby(grp_cols):
        g = g.sort_values("week_index")
        ids = g["competency_id"].tolist()
        for i in range(len(ids) - 1):
            edges.append({
                "source_competency_id": ids[i],
                "relation": "prerequisiteOf",
                "target_competency_id": ids[i + 1],
                "code_module": mod,
                "code_presentation": pres
            })
    return pd.DataFrame(edges)


def write_prereq_edges_from_competencies(
    competencies_path: Path | None = None, out_path: Path | None = None
) -> Path:
    """
    Read competencies.csv and write edges.csv with week-to-week prerequisite edges.
    """

    if competencies_path is None:
        competencies_path = get_data_path("processed/competencies.csv")
    competencies = pd.read_csv(competencies_path)
    edges = build_prereq_edges(competencies)

    if out_path is None:
        out_path = get_data_path("processed/edges.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    write_week_competencies_from_raw()
    write_prereq_edges_from_competencies()
