from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.cohort_exchange import (
    BenchmarkConfig, _checkpoint_signature, _completed_run_matches,
    _predict_evaluation_union, _prepare_checkpoint_dir, fold_assignment_table,
    run_cohort_exchange, write_protocol,
)
from src.run_provenance import (
    MANIFEST_RELATIVE_PATH, collect_run_provenance, sha256_file,
    validate_temporal_manifest,
)


def _fixture(root: Path) -> dict:
    source = root / "data/raw/Oulab/studentAssessment.csv"
    source.parent.mkdir(parents=True)
    source.write_text("id_assessment,date_submitted\n1,2\n", encoding="utf-8")
    processed = root / "data/processed/oulab"
    processed.mkdir(parents=True)
    outputs = {}
    for name in ("assess_weekly_evidence.csv", "vle_weekly_evidence.csv"):
        path = processed / name
        path.write_text("id_student,week_index\n1,1\n", encoding="utf-8")
        outputs[path.relative_to(root).as_posix()] = sha256_file(path)
    code = root / "src/example.py"
    code.parent.mkdir()
    code.write_text("POLICY = 'submission_time_v1'\n", encoding="utf-8")
    manifest = {
        "schema": 1, "policy": "submission_time_v1",
        "source_files": {source.relative_to(root).as_posix(): sha256_file(source)},
        "output_files": outputs,
        "policy_details": {"day_window": "0 <= day < 7 * week"},
    }
    (root / MANIFEST_RELATIVE_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class TemporalManifestTests(unittest.TestCase):
    def test_missing_manifest_fails_with_rebuild_command(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "python3 scripts/rebuild_oulad_temporal.py"):
                validate_temporal_manifest(Path(directory))

    def test_kdd_does_not_require_oulad_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(validate_temporal_manifest(Path(directory), "kdd"))
            result = collect_run_provenance("kdd", (1,), Path(directory))
            self.assertIsNone(result["temporal_policy"])

    def test_valid_manifest_then_modified_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _fixture(root)
            self.assertEqual(validate_temporal_manifest(root), expected)
            (root / next(iter(expected["output_files"]))).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed output_files"):
                validate_temporal_manifest(root)

    def test_modified_raw_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            (root / next(iter(manifest["source_files"]))).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed source_files"):
                validate_temporal_manifest(root)

    def test_fingerprints_change_for_added_optional_features_and_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            first = collect_run_provenance("oulab", (2,), root)
            (root / "data/processed/oulab/static_features.csv").write_text("new", encoding="utf-8")
            second = collect_run_provenance("oulab", (2,), root)
            self.assertNotEqual(first["input_files"], second["input_files"])
            (root / "src/example.py").write_text("POLICY = 'changed'\n", encoding="utf-8")
            third = collect_run_provenance("oulab", (2,), root)
            self.assertNotEqual(second["source_code"], third["source_code"])

    def test_unvalidated_landmark_features_cannot_enter_corrected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            (root / "data/processed/oulab/behavioral_week18.csv").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not cover behavioral_week18.csv"):
                collect_run_provenance("oulab", (18,), root)


class ResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = BenchmarkConfig(
            dataset="oulab", weeks=(2,), models=("flat_hgb",),
            folds=2, repeats=1, provenance={"schema": 1, "input_files": {"data.csv": "a"}},
        )

    def _completed(self, directory: Path, config: BenchmarkConfig | None = None):
        for name in ("predictions.csv.gz", "cohort_membership.csv.gz", "cohort_composition.csv", "fold_assignments.csv.gz"):
            (directory / name).write_bytes(b"fixture")
        write_protocol(self.config if config is None else config, directory)

    def test_signature_covers_split_weeks_models_and_provenance(self):
        baseline = _checkpoint_signature(self.config)
        for changed in (
            replace(self.config, split_unit="learner"),
            replace(self.config, weeks=(2, 4)),
            replace(self.config, models=("temporal_hgb",)),
            replace(self.config, provenance={"schema": 1, "input_files": {"data.csv": "b"}}),
        ):
            self.assertNotEqual(baseline, _checkpoint_signature(changed))

    def test_checkpoint_refuses_changed_inputs_even_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _prepare_checkpoint_dir(self.config, root, resume=True)
            changed = replace(self.config, provenance={"changed": True})
            for resume in (True, False):
                with self.assertRaisesRegex(ValueError, "new --output"):
                    _prepare_checkpoint_dir(changed, root, resume=resume)

    def test_orphan_checkpoint_is_not_silently_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            checkpoint = root / "checkpoints/week_002__flat_hgb.csv.gz"
            checkpoint.write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "no provenance"):
                _prepare_checkpoint_dir(self.config, root, resume=True)
            self.assertEqual(checkpoint.read_bytes(), b"old")

    def test_completed_run_requires_matching_provenance_split_and_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._completed(root)
            self.assertTrue(_completed_run_matches(self.config, root))
            self.assertFalse(_completed_run_matches(replace(self.config, split_unit="learner"), root))
            self.assertFalse(_completed_run_matches(replace(self.config, provenance={"new": 1}), root))
            (root / "predictions.csv.gz").write_bytes(b"changed")
            self.assertFalse(_completed_run_matches(self.config, root))
            with self.assertRaisesRegex(ValueError, "final predictions"):
                _prepare_checkpoint_dir(self.config, root, resume=True)

    def test_legacy_completed_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._completed(root)
            path = root / "protocol.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["provenance"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final predictions"):
                _prepare_checkpoint_dir(self.config, root, resume=False)

    def test_augmentation_rejects_changed_provenance_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._completed(source, replace(self.config, provenance={"old": True}))
            with patch("src.cohort_exchange.fold_assignment_table", return_value=pd.DataFrame({"fold": [1]})), \
                    patch("src.cohort_exchange.run_landmark_benchmark") as fit:
                with self.assertRaisesRegex(ValueError, "augmentation source settings"):
                    run_cohort_exchange(self.config, root / "new", augment_from=source, fit_only=True)
                fit.assert_not_called()

    def test_protocol_records_split_provenance_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._completed(root)
            protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["split_unit"], "presentation")
            self.assertEqual(protocol["provenance"], self.config.provenance)
            self.assertEqual(protocol["artifacts"]["fold_assignments.csv.gz"], sha256_file(root / "fold_assignments.csv.gz"))


class PairedPredictionTests(unittest.TestCase):
    def test_batch_dependent_estimator_is_called_once_for_overlap(self):
        class BatchDependentEstimator:
            calls = 0

            def predict_proba(self, features):
                self.calls += 1
                success = features["x"].to_numpy() * 0.01 + len(features) * 0.1
                return np.column_stack((1 - success, success))

        estimator = BatchDependentEstimator()
        base = pd.DataFrame({
            "x": [1, 2, 3, 4], "_fold": [1, 1, 1, 2],
            "activity_conditioned": [True, True, False, True],
            "cutoff_valid": [False, True, True, True],
        }, index=[10, 20, 30, 40])
        scores = _predict_evaluation_union(estimator, base, ["x"], (
            ("AA", "activity_conditioned"), ("AV", "cutoff_valid")
        ), 1)
        self.assertEqual(estimator.calls, 1)
        self.assertAlmostEqual(scores.loc[20], 0.32)
        self.assertTrue(np.isnan(scores.loc[40]))
        aa = scores[base.activity_conditioned & base._fold.eq(1)]
        av = scores[base.cutoff_valid & base._fold.eq(1)]
        self.assertEqual(aa.loc[20], av.loc[20])

    def test_learner_fold_assignments_are_reproducible_across_enrolments(self):
        roster = pd.DataFrame([
            [learner, module, "P1", learner % 2]
            for learner in range(1, 13) for module in ("A", "B")
        ], columns=["id_student", "code_module", "code_presentation", "label"])
        config = BenchmarkConfig("oulab", (2,), ("flat_hgb",), folds=3, repeats=2, split_unit="learner")
        with patch("src.cohort_exchange.load_roster", return_value=roster):
            first = fold_assignment_table(config)
            second = fold_assignment_table(config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), len(roster) * 2)
        self.assertTrue(first.groupby(["repeat", "id_student"])["fold"].nunique().eq(1).all())


if __name__ == "__main__":
    unittest.main()
