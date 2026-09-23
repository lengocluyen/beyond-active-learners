from __future__ import annotations

import subprocess
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.rebuild_oulad_temporal import validate_rebuilt_tables, WEEKS
from scripts.run_tlt_corrected import commands, parse_args, run_pipeline


def fixtures():
    roster = pd.DataFrame([[1, "A", "P"], [2, "A", "P"]],
                          columns=["id_student", "code_module", "code_presentation"])
    # Assessment 10 is due before week 2 but is submitted exactly at its cutoff.
    submissions = pd.DataFrame([
        [1, 10, 14, 0, 80], [1, 11, 1, 0, 60],
        [2, 12, -1, 0, 99], [2, 13, 2, 1, 98],
    ], columns=["id_student", "id_assessment", "date_submitted", "is_banked", "score"])
    assessments = pd.DataFrame([[i, "A", "P", 5] for i in range(10, 14)],
                               columns=["id_assessment", "code_module", "code_presentation", "date"])
    events = pd.DataFrame([[2, "A", "P", -1, 100], [2, "A", "P", 0, 3],
                           [2, "A", "P", 14, 5]],
                          columns=["id_student", "code_module", "code_presentation", "date", "sum_click"])
    assessment_table = pd.DataFrame([[1, "A", "P", 1, 1, .6, .6], [1, "A", "P", 3, 1, .8, .8]],
        columns=["id_student", "code_module", "code_presentation", "week_index",
                 "assess_attempts", "assess_score_mean", "assess_score_max"])
    vle_table = pd.DataFrame([[2, "A", "P", 1, 3, 1], [2, "A", "P", 3, 5, 1]],
        columns=["id_student", "code_module", "code_presentation", "week_index", "clicks_total", "active_days"])
    behavioral = {week: roster.assign(time_submissions=[1 if week == 2 else 2, 0]) for week in WEEKS}
    return submissions, assessments, events, assessment_table, vle_table, roster, behavioral


class RawTimestampAuditTests(unittest.TestCase):
    def test_excludes_prestart_banked_and_cutoff_submission(self):
        result = validate_rebuilt_tables(*fixtures())
        self.assertEqual(result["excluded_assessment_rows"], 2)
        self.assertEqual(result["excluded_vle_rows"], 1)
        self.assertEqual(result["landmarks"][0]["admissible_submissions"], 1)
        self.assertEqual(result["landmarks"][1]["admissible_submissions"], 2)
        self.assertEqual(result["landmarks"][0]["vle_clicks"], 3)

    def test_rejects_scheduled_date_assignment(self):
        data = list(fixtures())
        data[3].loc[1, "week_index"] = 2
        with self.assertRaisesRegex(ValueError, "weekly group keys"):
            validate_rebuilt_tables(*data)

    def test_rejects_prestart_click_contamination(self):
        data = list(fixtures())
        data[4].loc[0, "clicks_total"] = 103
        with self.assertRaisesRegex(ValueError, "clicks_total"):
            validate_rebuilt_tables(*data)

    def test_rejects_stale_behavioral_submission_counts(self):
        data = list(fixtures())
        data[6][2].loc[1, "time_submissions"] = 2
        with self.assertRaisesRegex(ValueError, "behavioral submission counts"):
            validate_rebuilt_tables(*data)

    def test_nonfinite_timestamps_are_excluded(self):
        data = list(fixtures())
        invalid_submissions = data[0].iloc[[0, 0, 0]].copy()
        invalid_submissions["date_submitted"] = [float("inf"), -float("inf"), float("nan")]
        data[0] = pd.concat([data[0], invalid_submissions], ignore_index=True)
        invalid_events = data[2].iloc[[0, 0, 0]].copy()
        invalid_events["date"] = [float("inf"), -float("inf"), float("nan")]
        data[2] = pd.concat([data[2], invalid_events], ignore_index=True)
        result = validate_rebuilt_tables(*data)
        self.assertEqual(result["excluded_assessment_rows"], 5)
        self.assertEqual(result["excluded_vle_rows"], 4)


class LauncherTests(unittest.TestCase):
    def test_main_and_learner_protocols_are_separate(self):
        stages = dict(commands(Path("fresh_results"), 3, "python3"))
        self.assertEqual(stages["main"][0], "python3")
        self.assertIn("--reference-baselines", stages["main"])
        self.assertNotIn("--no-hazard", stages["main"])
        self.assertIn("--no-hazard", stages["learner_disjoint"])
        self.assertEqual(stages["main"][stages["main"].index("--split-unit") + 1], "presentation")
        self.assertEqual(stages["learner_disjoint"][stages["learner_disjoint"].index("--split-unit") + 1], "learner")
        for stage in ("main", "learner_disjoint"):
            self.assertNotIn("--augment-from", stages[stage])
            self.assertEqual(stages[stage][stages[stage].index("--bootstrap") + 1], "2000")

    def test_failed_preparation_prevents_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-root", temporary])
            with patch("scripts.run_tlt_corrected.run_logged",
                       side_effect=subprocess.CalledProcessError(2, ["prepare"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_pipeline(args)
            self.assertEqual(run.call_count, 1)
            self.assertIn("scripts/rebuild_oulad_temporal.py", run.call_args.args[0])

    def test_skip_rebuild_still_requires_manifest_and_prepare_only_never_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-root", temporary, "--skip-rebuild", "--prepare-only"])
            with patch("scripts.run_tlt_corrected.run_logged") as run, \
                 patch("src.run_provenance.validate_temporal_manifest", return_value={"schema": 1}) as validate:
                run_pipeline(args)
            validate.assert_called_once()
            run.assert_not_called()

    def test_invalid_manifest_prevents_fitting_with_skip_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-root", temporary, "--skip-rebuild"])
            with patch("scripts.run_tlt_corrected.run_logged") as run, \
                 patch("src.run_provenance.validate_temporal_manifest", side_effect=ValueError("stale inputs")):
                with self.assertRaisesRegex(ValueError, "stale inputs"):
                    run_pipeline(args)
            run.assert_not_called()

    def test_unknown_preprocessing_policy_fails_with_skip_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "data/processed/oulab/temporal_preprocessing.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"schema": 1, "policy": "unverified_old_pipeline"}), encoding="utf-8")
            args = parse_args(["--output-root", str(root / "results"), "--skip-rebuild"])
            with patch("scripts.run_tlt_corrected.PROJECT_ROOT", root), \
                 patch("scripts.run_tlt_corrected.run_logged") as run:
                with self.assertRaisesRegex(ValueError, "unsupported temporal preprocessing"):
                    run_pipeline(args)
            run.assert_not_called()

    def test_main_runner_rejection_stops_later_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-root", temporary, "--skip-rebuild"])
            with patch("src.run_provenance.validate_temporal_manifest", return_value={"schema": 1}), \
                 patch("scripts.run_tlt_corrected.run_logged",
                       side_effect=subprocess.CalledProcessError(2, ["run_cohort_exchange.py"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_pipeline(args)
            self.assertEqual(run.call_count, 1)
            self.assertIn("--reference-baselines", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
