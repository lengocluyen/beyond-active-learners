from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.evidence_mapping import (
    admissible_assessment_submissions,
    build_assess_weekly_evidence,
    build_vle_weekly_evidence,
)
from src.features_behavioral import cohort_features, gap_features, timing_features
from src.pcg_ut import _load_weekly_events


def metadata() -> pd.DataFrame:
    return pd.DataFrame({
        'id_assessment': [1, 2, 3, 4],
        'code_module': ['A'] * 4, 'code_presentation': ['P'] * 4,
        'date': [2, 30, 4, 40], 'weight': [25] * 4,
    })


def submissions(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        'id_assessment', 'id_student', 'date_submitted', 'is_banked', 'score'
    ])


class AssessmentTimingTests(unittest.TestCase):
    def test_actual_submission_day_and_strict_cutoff(self):
        evidence = build_assess_weekly_evidence(submissions([
            [1, 11, 15, 0, 100],  # due 2, submitted after cutoff 14
            [2, 12, 7, 0, 60],    # submitted early, before deadline 30
            [3, 13, 14, 0, 90],   # precisely at cutoff, excluded at week 2
        ]), metadata())
        self.assertEqual(evidence.set_index('id_student').week_index.to_dict(),
                         {11: 3, 12: 2, 13: 3})
        self.assertEqual(set(evidence.loc[evidence.week_index.le(2), 'id_student']), {12})

    def test_future_submission_cannot_change_an_earlier_snapshot(self):
        before = submissions([[1, 11, 1, 0, 40]])
        future = submissions([[3, 11, 14, 0, 100]])
        a = build_assess_weekly_evidence(before, metadata())
        b = build_assess_weekly_evidence(pd.concat([before, future]), metadata())
        assert_frame_equal(a, b.loc[b.week_index.le(2)].reset_index(drop=True))

    def test_scheduled_deadline_is_not_evidence_time(self):
        records = submissions([[1, 11, 7, 0, 50], [2, 11, 13, 0, 80]])
        changed = metadata()
        changed['date'] = [400, np.nan, -100, 0]
        assert_frame_equal(build_assess_weekly_evidence(records, metadata()),
                           build_assess_weekly_evidence(records, changed))

    def test_no_deadline_fallback_banked_or_prestart_activity(self):
        records = submissions([
            [1, 11, np.nan, 0, 80], [1, 12, -1, 0, 80],
            [1, 13, 0, 1, 80], [1, 14, 0, 0, 80],
            [1, 15, np.inf, 0, 80], [1, 16, 1, np.nan, 80],
        ])
        result = build_assess_weekly_evidence(records, metadata())
        self.assertEqual(result.id_student.tolist(), [14])
        self.assertEqual(result.week_index.tolist(), [1])

    def test_missing_event_clock_fails(self):
        records = submissions([[1, 11, 1, 0, 80]]).drop(columns='date_submitted')
        with self.assertRaisesRegex(ValueError, 'date_submitted'):
            build_assess_weekly_evidence(records, metadata())

    def test_missing_banking_status_fails(self):
        records = submissions([[1, 11, 1, 0, 80]]).drop(columns='is_banked')
        with self.assertRaisesRegex(ValueError, 'is_banked'):
            admissible_assessment_submissions(records)

    def test_unmatched_metadata_fails(self):
        with self.assertRaisesRegex(ValueError, 'metadata'):
            build_assess_weekly_evidence(submissions([[99, 11, 1, 0, 80]]), metadata())

    def test_empty_admissible_evidence_is_supported(self):
        result = build_assess_weekly_evidence(submissions([[1, 11, -1, 0, 80]]), metadata())
        self.assertTrue(result.empty)
        self.assertIn('assess_attempts', result)

    def test_weekly_loader_obeys_round_trip_cutoff(self):
        records = submissions([[1, 11, 1, 0, 40], [3, 12, 14, 0, 100]])
        evidence = build_assess_weekly_evidence(records, metadata())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence.to_csv(root/'assess_weekly_evidence.csv', index=False)
            pd.DataFrame(columns=['id_student', 'code_module', 'code_presentation',
                                  'week_index', 'clicks_total']).to_csv(root/'vle_weekly_evidence.csv', index=False)
            with patch('src.pcg_ut.get_data_path', side_effect=lambda p: root/Path(p).name):
                early, later = _load_weekly_events(2), _load_weekly_events(3)
        self.assertEqual(set(early.id_student), {11})
        self.assertEqual(set(later.id_student), {11, 12})


class VLETimingTests(unittest.TestCase):
    def test_negative_days_are_not_folded_into_week_one(self):
        events = pd.DataFrame({
            'id_student': [11, 12, 12], 'code_module': ['A'] * 3,
            'code_presentation': ['P'] * 3, 'id_site': [1] * 3,
            'date': [-1, 0, 14], 'sum_click': [500, 2, 3],
        })
        events['code_module'] = pd.Categorical(events.code_module, categories=['A', 'B'])
        sites = pd.DataFrame({'id_site': [1], 'activity_type': ['resource']})
        evidence = build_vle_weekly_evidence(events, sites)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(set(evidence.id_student), {12})
        self.assertEqual(evidence.set_index('week_index').clicks_total.to_dict(), {1: 2, 3: 3})


class BehavioralTimingTests(unittest.TestCase):
    def test_direct_vle_inputs_exclude_prestart_and_cutoff_events(self):
        events = pd.DataFrame({
            'id_student': [11, 12, 12, 13], 'code_module': ['A'] * 4,
            'code_presentation': ['P'] * 4, 'id_site': [1] * 4,
            'date': [-1, 0, 14, 3], 'sum_click': [500, 2, 900, 4],
        })
        clean = events.loc[events.date.ge(0) & events.date.lt(14)]
        for builder in (gap_features, cohort_features):
            with self.subTest(builder=builder.__name__):
                assert_frame_equal(builder(2, events), builder(2, clean))

    def test_same_cutoff_rule_and_early_submissions_do_not_cancel_missed_due_work(self):
        records = submissions([
            [2, 11, 1, 0, 70],   # early submission to deadline 30
            [1, 11, 14, 0, 80],  # due 2 but cutoff submission not yet observed
            [3, 12, -1, 0, 90], [1, 12, 0, 1, 90],
        ])
        roster = pd.DataFrame({'id_student': [11, 12], 'code_module': ['A', 'A'],
                               'code_presentation': ['P', 'P']})
        frames = {'studentAssessment.csv': records, 'assessments.csv': metadata(),
                  'studentInfo.csv': roster}
        with patch('src.features_behavioral.pd.read_csv', side_effect=lambda p: frames[Path(p).name].copy()):
            result = timing_features(2).set_index('id_student')
        self.assertEqual(result.loc[11, 'time_submissions'], 1)
        self.assertEqual(result.loc[11, 'time_missed'], 2)
        self.assertEqual(result.loc[12, 'time_submissions'], 0)
        self.assertEqual(result.loc[12, 'time_missed'], 2)


if __name__ == '__main__':
    unittest.main()
