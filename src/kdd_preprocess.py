from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import get_data_path


def _kdd_root() -> Path:
    return get_data_path("raw/KDDCup2015")


def _processed_dir() -> Path:
    return get_data_path("processed")


def _read_kdd_enrollments() -> pd.DataFrame:
    root = _kdd_root()
    train = pd.read_csv(root / "train" / "enrollment_train.csv")
    test = pd.read_csv(root / "test" / "enrollment_test.csv")
    df = pd.concat([train, test], ignore_index=True)
    return df


def _read_kdd_truth() -> pd.DataFrame:
    root = _kdd_root()
    train = pd.read_csv(root / "train" / "truth_train.csv", header=None, names=["enrollment_id", "dropout"])
    test = pd.read_csv(root / "test" / "truth_test.csv", header=None, names=["enrollment_id", "dropout"])
    df = pd.concat([train, test], ignore_index=True)
    return df


def _read_kdd_logs() -> pd.DataFrame:
    root = _kdd_root()
    train = pd.read_csv(root / "train" / "log_train.csv")
    test = pd.read_csv(root / "test" / "log_test.csv")
    df = pd.concat([train, test], ignore_index=True)
    return df


def _course_start_dates() -> pd.DataFrame:
    # date.csv: course_id,from,to (YYYY-MM-DD)
    dates = pd.read_csv(_kdd_root() / "date.csv")
    dates["from"] = pd.to_datetime(dates["from"], errors="coerce")
    return dates.rename(columns={"from": "start_date"})


def build_kdd_labels(label_mode: str = "certificate") -> pd.DataFrame:
    enroll = _read_kdd_enrollments()
    truth = _read_kdd_truth()
    df = enroll.merge(truth, on="enrollment_id", how="left")
    if label_mode != "certificate":
        raise ValueError("KDD label_mode supports only 'certificate' for now.")

    # dropout=1 means no certificate; we flip to certificate-completed label
    df["label"] = (1 - df["dropout"].fillna(1)).astype(int)
    out = df.rename(
        columns={
            "enrollment_id": "id_student",
            "course_id": "code_module",
        }
    )[["id_student", "code_module", "label"]]
    out["code_presentation"] = out["code_module"]
    out = out[["id_student", "code_module", "code_presentation", "label"]]
    _processed_dir().mkdir(parents=True, exist_ok=True)
    out.to_csv(_processed_dir() / "labels.csv", index=False)
    return out


def _week_index_from_time(time_series: pd.Series, start_date: pd.Series) -> pd.Series:
    dt = pd.to_datetime(time_series, errors="coerce")
    delta_days = (dt - start_date).dt.days
    week = (delta_days // 7) + 1
    return week


def build_kdd_weekly_evidence() -> pd.DataFrame:
    enroll = _read_kdd_enrollments()
    logs = _read_kdd_logs()
    dates = _course_start_dates()

    logs = logs.merge(enroll, on="enrollment_id", how="left")
    logs = logs.merge(dates[["course_id", "start_date"]], on="course_id", how="left")
    logs["week_index"] = _week_index_from_time(logs["time"], logs["start_date"])
    logs = logs.dropna(subset=["week_index"])
    logs["week_index"] = logs["week_index"].astype(int)
    logs = logs[logs["week_index"] >= 1]

    logs["day"] = pd.to_datetime(logs["time"], errors="coerce").dt.date
    base = (
        logs.groupby(["enrollment_id", "course_id", "week_index"], as_index=False)
        .agg(clicks_total=("event", "count"), active_days=("day", "nunique"))
    )

    src_counts = (
        logs.pivot_table(
            index=["enrollment_id", "course_id", "week_index"],
            columns="source",
            values="event",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    src_cols = [c for c in src_counts.columns if c not in {"enrollment_id", "course_id", "week_index"}]
    src_counts = src_counts.rename(columns={c: f"clicks_src_{c}" for c in src_cols})

    out = base.merge(src_counts, on=["enrollment_id", "course_id", "week_index"], how="left")
    out = out.rename(
        columns={
            "enrollment_id": "id_student",
            "course_id": "code_module",
        }
    )
    out["code_presentation"] = out["code_module"]
    _processed_dir().mkdir(parents=True, exist_ok=True)
    out.to_csv(_processed_dir() / "vle_weekly_evidence.csv", index=False)
    return out


def build_kdd_assess_weekly_evidence() -> pd.DataFrame:
    # KDD Cup 2015 does not include assessment scores; create empty evidence aligned to weeks.
    vle = pd.read_csv(_processed_dir() / "vle_weekly_evidence.csv")
    key_cols = ["id_student", "code_module", "code_presentation", "week_index"]
    out = vle[key_cols].drop_duplicates()
    out["assess_attempts"] = 0
    out["assess_score_mean"] = 0.0
    out["assess_score_max"] = 0.0
    out["assess_score_weighted"] = 0.0
    out.to_csv(_processed_dir() / "assess_weekly_evidence.csv", index=False)
    return out


def build_kdd_competencies() -> pd.DataFrame:
    dates = pd.read_csv(_kdd_root() / "date.csv")
    dates["from"] = pd.to_datetime(dates["from"], errors="coerce")
    dates["to"] = pd.to_datetime(dates["to"], errors="coerce")
    rows = []
    for _, row in dates.iterrows():
        if pd.isna(row["from"]) or pd.isna(row["to"]):
            continue
        weeks = int(((row["to"] - row["from"]).days // 7) + 1)
        weeks = max(1, weeks)
        for w in range(1, weeks + 1):
            rows.append(
                {
                    "competency_id": f"{row['course_id']}_week{w}",
                    "code_module": row["course_id"],
                    "code_presentation": row["course_id"],
                    "week_index": w,
                    "name": f"{row['course_id']} Week {w}",
                }
            )
    out = pd.DataFrame(rows)
    _processed_dir().mkdir(parents=True, exist_ok=True)
    out.to_csv(_processed_dir() / "competencies.csv", index=False)
    return out


def build_kdd_edges() -> pd.DataFrame:
    comp = pd.read_csv(_processed_dir() / "competencies.csv")
    rows = []
    for mod, group in comp.groupby("code_module"):
        group = group.sort_values("week_index")
        ids = list(group["competency_id"].values)
        pres = str(group["code_presentation"].iloc[0]) if "code_presentation" in group.columns and len(group) > 0 else mod
        for i in range(len(ids) - 1):
            rows.append(
                {
                    "source_competency_id": ids[i],
                    "relation": "prerequisiteOf",
                    "target_competency_id": ids[i + 1],
                    "code_module": mod,
                    "code_presentation": pres,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(_processed_dir() / "edges.csv", index=False)
    return out

def write_kdd_processed(label_mode: str = "certificate") -> None:
    build_kdd_labels(label_mode=label_mode)
    build_kdd_competencies()
    build_kdd_edges()
    build_kdd_weekly_evidence()
    build_kdd_assess_weekly_evidence()


