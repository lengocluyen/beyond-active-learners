from __future__ import annotations

import pandas as pd

from .paths import get_data_path


def make_labels(label_mode: str = "binary") -> pd.DataFrame:
    """
    Build pass/fail labels from OULAD studentInfo final_result.

    Mapping (documented here for traceability):
    - label_mode="binary":
        Pass, Distinction -> 1
        Fail, Withdrawn  -> 0
    - label_mode="multiclass4":
        Withdrawn -> 0
        Fail      -> 1
        Pass      -> 2
        Distinction -> 3
    Missing/unknown final_result are dropped.
    """

    # Load raw OULAD studentInfo (expects the canonical OULAD CSV name).
    student_info_path = get_data_path("raw/studentInfo.csv")
    df = pd.read_csv(student_info_path)

    # Only keep the columns we need to produce labels.
    cols = ["id_student", "code_module", "code_presentation", "final_result"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"studentInfo missing columns: {missing_cols}")
    df = df[cols]

    if label_mode == "binary":
        mapping = {
            "Pass": 1,
            "Distinction": 1,
            "Fail": 0,
            "Withdrawn": 0,
        }
    elif label_mode == "multiclass4":
        mapping = {
            "Withdrawn": 0,
            "Fail": 1,
            "Pass": 2,
            "Distinction": 3,
        }
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    df["label_name"] = df["final_result"].astype(str)
    df["label"] = df["final_result"].map(mapping)

    # Drop rows with missing/unknown outcomes.
    df = df.dropna(subset=["label"]).drop(columns=["final_result"])
    df["label"] = df["label"].astype(int)

    # Write labels to processed data folder.
    out_path = get_data_path("processed/labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    make_labels()

