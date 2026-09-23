"""Rebuild and independently check OULAD event-time features before fitting.

Run from the project checkout: python3 scripts/rebuild_oulad_temporal.py
Existing evidence is backed up; model fitting is never performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEKS = (2, 4, 6, 8, 10, 12, 14, 16)
KEY = ["id_student", "code_module", "code_presentation"]
WEEK_KEY = KEY + ["week_index"]
POLICY = "submission_time_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def required_inputs() -> tuple[dict[str, Path], dict[str, Path]]:
    """Resolve every dependency before any derived artifact is replaced."""
    os.environ["PCG_DATASET"] = "oulab"
    from src.paths import get_data_path

    raw = {
        name: get_data_path(f"raw/{name}.csv")
        for name in (
            "studentAssessment", "assessments", "studentVle", "vle",
            "studentInfo", "studentRegistration",
        )
    }
    retained = {
        name: get_data_path(f"processed/{name}.csv")
        for name in ("competencies", "labels", "static_features")
    }
    retained.update({
        f"traversal_week{week}": get_data_path(f"processed/traversal_week{week}.csv")
        for week in WEEKS
    })
    missing = [str(path) for path in [*raw.values(), *retained.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required existing OULAD inputs are missing; no outputs were replaced:\n  "
            + "\n  ".join(missing)
        )
    return raw, retained


def _compare_grouped(expected, actual, columns: list[str], name: str) -> None:
    import numpy as np
    import pandas as pd

    left = expected.set_index(WEEK_KEY)[columns].sort_index()
    right = actual.set_index(WEEK_KEY)[columns].sort_index()
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError(f"{name}: duplicate learner-week groups")
    # CSV roundtrips change category dtypes but must not change the actual keys.
    if set(map(tuple, left.index)) != set(map(tuple, right.index)):
        raise ValueError(f"{name}: weekly group keys disagree with independent raw timestamps")
    right = right.reindex(left.index)
    for column in columns:
        l = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
        r = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(l, r, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"{name}: {column} disagrees with independent raw timestamps")


def validate_rebuilt_tables(raw_submissions, assessments, raw_vle, assessment_table,
                            vle_table, roster, behavioral: dict) -> dict:
    """Audit builders using raw timestamps and an independent aggregation.

    Deliberately does not use the production date-to-week/admissibility helpers.
    Scheduled assessment dates are never consulted by this reconstruction.
    """
    import numpy as np
    import pandas as pd

    submitted = pd.to_numeric(raw_submissions["date_submitted"], errors="coerce")
    banked = pd.to_numeric(raw_submissions["is_banked"], errors="coerce")
    keep = np.isfinite(submitted) & submitted.ge(0) & banked.eq(0)
    sa = raw_submissions.loc[keep].copy()
    sa["date_submitted"] = submitted.loc[keep]
    sa = sa.merge(assessments[["id_assessment", "code_module", "code_presentation"]],
                  on="id_assessment", validate="many_to_one")
    if len(sa) != int(keep.sum()):
        raise ValueError("Assessment metadata missing for admissible submissions")
    sa["week_index"] = np.floor(sa["date_submitted"] / 7).astype(int) + 1
    sa["_score"] = pd.to_numeric(sa["score"], errors="coerce") / 100
    expected_assess = sa.groupby(WEEK_KEY, observed=True).agg(
        assess_attempts=("id_assessment", "count"),
        assess_score_mean=("_score", "mean"),
        assess_score_max=("_score", "max"),
    ).reset_index()
    _compare_grouped(expected_assess, assessment_table,
                     ["assess_attempts", "assess_score_mean", "assess_score_max"], "assessment")

    dates = pd.to_numeric(raw_vle["date"], errors="coerce")
    vle_keep = np.isfinite(dates) & dates.ge(0)
    sv = raw_vle.loc[vle_keep, KEY + ["date", "sum_click"]].copy()
    sv["week_index"] = np.floor(pd.to_numeric(sv["date"]) / 7).astype(int) + 1
    expected_vle = sv.groupby(WEEK_KEY, observed=True).agg(
        clicks_total=("sum_click", "sum"), active_days=("date", "nunique")
    ).reset_index()
    _compare_grouped(expected_vle, vle_table, ["clicks_total", "active_days"], "VLE")

    roster_keys = set(map(tuple, roster[KEY].drop_duplicates().to_numpy()))
    landmarks = []
    for week in WEEKS:
        horizon = 7 * week
        actual_assess = assessment_table[assessment_table.week_index.between(1, week)]
        actual_vle = vle_table[vle_table.week_index.between(1, week)]
        cutoff_submissions = sa[sa.date_submitted.lt(horizon)]
        expected_vle_window = expected_vle[expected_vle.week_index.le(week)]
        expected_active = set(map(tuple, cutoff_submissions[KEY].to_numpy()))
        expected_active.update(map(tuple, expected_vle_window[KEY].to_numpy()))
        actual_active = set(map(tuple, actual_assess[KEY].to_numpy()))
        actual_active.update(map(tuple, actual_vle[KEY].to_numpy()))
        if expected_active != actual_active:
            raise ValueError(f"Week {week}: activity membership differs from raw timestamps")
        if int(actual_assess.assess_attempts.sum()) != len(cutoff_submissions):
            raise ValueError(f"Week {week}: submission count crosses its cutoff")

        # Behavioral timing must use the same submission admissibility rule.
        frame = behavioral[week]
        if frame.duplicated(KEY).any():
            raise ValueError(f"Week {week}: duplicated behavioral roster records")
        if set(map(tuple, frame[KEY].to_numpy())) != roster_keys:
            raise ValueError(f"Week {week}: behavioral table does not cover the roster")
        expected_counts = cutoff_submissions.groupby(KEY, observed=True).size()
        actual_counts = frame.set_index(KEY)["time_submissions"]
        if not np.allclose(actual_counts.to_numpy(),
                           expected_counts.reindex(actual_counts.index).fillna(0).to_numpy()):
            raise ValueError(f"Week {week}: behavioral submission counts disagree")
        landmarks.append({
            "week": week, "cutoff_day": horizon,
            "admissible_submissions": len(cutoff_submissions),
            "activity_enrolments": len(actual_active & roster_keys),
            "vle_clicks": int(actual_vle.clicks_total.sum()),
            "future_assessment_records": 0,
            "activity_membership_matches_raw": True,
        })
    return {
        "weekly_assessment_groups": len(expected_assess),
        "weekly_vle_groups": len(expected_vle),
        "assessment_attempts_and_scores_match_raw": True,
        "vle_clicks_and_active_days_match_raw": True,
        "excluded_assessment_rows": int((~keep).sum()),
        "excluded_vle_rows": int((~vle_keep).sum()),
        "landmarks": landmarks,
    }


def rebuild() -> Path:
    import pandas as pd
    from src.evidence_mapping import build_assess_weekly_evidence, build_vle_weekly_evidence
    from src.features_behavioral import build_behavioral_features

    raw, retained = required_inputs()
    processed = retained["competencies"].parent
    manifest_path = processed / "temporal_preprocessing.json"
    sources = [*raw.values(), PROJECT_ROOT / "src/evidence_mapping.py",
               PROJECT_ROOT / "src/features_behavioral.py"]
    source_hashes = {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in sources}
    retained_hashes = {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in retained.values()}
    stage = Path(tempfile.mkdtemp(prefix="temporal_build_", dir=processed))
    print(f"[prepare] OULAD target: {processed}", flush=True)
    print(f"[prepare] Staging outputs in {stage}", flush=True)
    try:
        submissions = pd.read_csv(raw["studentAssessment"])
        assessments = pd.read_csv(raw["assessments"])
        competencies = pd.read_csv(retained["competencies"])
        roster = pd.read_csv(raw["studentInfo"])[KEY].drop_duplicates()
        print("[prepare] Reading raw VLE events with compact dtypes", flush=True)
        events = pd.read_csv(raw["studentVle"], dtype={
            "code_module": "category", "code_presentation": "category",
            "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32",
        })
        vle = pd.read_csv(raw["vle"])
        print("[prepare] Building submission-time assessment and VLE evidence", flush=True)
        assessment_table = build_assess_weekly_evidence(submissions, assessments, competencies)
        vle_table = build_vle_weekly_evidence(events, vle, competencies)
        generated = {
            "assess_weekly_evidence.csv": assessment_table,
            "vle_weekly_evidence.csv": vle_table,
        }
        behavioral = {}
        for week in WEEKS:
            print(f"[prepare] Building behavioral features: week {week}", flush=True)
            behavioral[week] = build_behavioral_features(week, events=events)
            generated[f"behavioral_week{week}.csv"] = behavioral[week]
        for name, frame in generated.items():
            frame.to_csv(stage / name, index=False)
        # Validate the serialized artifacts, not only their in-memory precursors.
        print("[prepare] Independent raw-timestamp validation at all eight landmarks", flush=True)
        validation = validate_rebuilt_tables(
            submissions, assessments, events,
            pd.read_csv(stage / "assess_weekly_evidence.csv"),
            pd.read_csv(stage / "vle_weekly_evidence.csv"), roster,
            {week: pd.read_csv(stage / f"behavioral_week{week}.csv") for week in WEEKS},
        )
        for relative, original_hash in {**source_hashes, **retained_hashes}.items():
            if sha256(PROJECT_ROOT / relative) != original_hash:
                raise RuntimeError(f"Input changed during preprocessing: {relative}; no outputs replaced")
        backup = processed / "temporal_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        for name in [*generated, manifest_path.name]:
            existing = processed / name
            if existing.exists():
                shutil.copy2(existing, backup / name)
        # Only validated outputs are promoted. The manifest is committed last.
        for name in generated:
            os.replace(stage / name, processed / name)
        output_hashes = dict(retained_hashes)
        output_hashes.update({(processed / name).relative_to(PROJECT_ROOT).as_posix():
                              sha256(processed / name) for name in generated})
        manifest = {
            "schema": 1, "policy": POLICY,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "weeks": list(WEEKS),
            "policy_details": {
                "landmark_window": "0 <= event_day < 7 * week (half-open [0, 7w))",
                "assessment_event_day": "date_submitted",
                "banked_assessments": "excluded (is_banked must equal 0)",
                "prestart_events": "excluded from assessment, VLE, and behavioral features",
                "grade_availability": "scores assumed available at submission; grade-release timestamps are not observed",
                "retained_features": "labels, static profiles, and traversal features preserved",
            },
            "source_files": source_hashes, "output_files": output_hashes,
            "validation": validation,
            "backup_directory": backup.relative_to(PROJECT_ROOT).as_posix(),
        }
        atomic_json(manifest_path, manifest)
        print(f"[prepare] PASSED. Manifest: {manifest_path}", flush=True)
        print(f"[prepare] Previous outputs preserved in {backup}", flush=True)
        return manifest_path
    finally:
        # Delete only the known, freshly created staging directory within processed.
        if stage.is_dir() and stage.resolve().parent == processed.resolve() and stage.name.startswith("temporal_build_"):
            shutil.rmtree(stage)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rebuild()


if __name__ == "__main__":
    main()
