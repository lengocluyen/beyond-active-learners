from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.aggregate_metric_helpers import (
    average_precision_tie_correction,
    weighted_cluster_metric_draws_exact,
)
from scripts.postprocess_ap_intervals import merged_risk_thresholds
from src.cohort_exchange import (
    _weighted_cluster_average_precision_draws,
    _weighted_cluster_metric_draws,
)


class AveragePrecisionTieCorrectionTests(unittest.TestCase):
    def test_adjacent_success_probabilities_merge_in_risk_score(self):
        high = 0.30666666666666664
        low = float(np.nextafter(high, 0))
        self.assertLess(low, high)
        self.assertEqual(1.0 - low, 1.0 - high)
        frame = pd.DataFrame(dict(cluster_id=["A", "A", "B", "B"],
                                  y=[0, 1, 0, 1], p_success=[low, high, low, high]))
        counts = np.array([[1, 1]])
        self.assertEqual(merged_risk_thresholds(frame), 1)
        order = np.array(["A", "B"])
        original = _weighted_cluster_metric_draws(frame, counts, order)
        corrected = weighted_cluster_metric_draws_exact(frame, counts, order)
        self.assertEqual(original["pr_auc_risk"][0], 0.5)
        self.assertEqual(original["auc"][0], 1.0)
        self.assertEqual(corrected["pr_auc_risk"][0], 0.5)
        self.assertEqual(average_precision_tie_correction(frame, counts, order)[0], -0.5)
        for metric in ("auc", "brier", "ece"):
            np.testing.assert_array_equal(corrected[metric], original[metric])

    def test_multiple_collapsed_blocks_match_sklearn_with_cluster_weights(self):
        high_a, high_b = 0.2, 0.30666666666666664
        low_a, low_b = float(np.nextafter(high_a, 0)), float(np.nextafter(high_b, 0))
        rows = []
        for cluster, labels in [("A", (0, 1, 1, 0, 0, 1)),
                                ("B", (1, 0, 0, 1, 0, 1)),
                                ("C", (0, 0, 1, 1, 1, 0))]:
            for y, p in zip(labels, (low_a, high_a, low_b, high_b, .7, .9)):
                rows.append(dict(cluster_id=cluster, y=y, p_success=p))
        frame = pd.DataFrame(rows).sample(frac=1, random_state=4)
        order = np.array(["C", "A", "B"])
        counts = np.array([[1, 1, 1], [3, 0, 0], [0, 1, 2], [2, 2, 0], [0, 0, 4]])
        original = _weighted_cluster_metric_draws(frame, counts, order)
        correction = average_precision_tie_correction(frame, counts, order, batch_size=2)
        expected, legacy = [], []
        for draw in counts:
            weights = frame.cluster_id.map(dict(zip(order, draw))).to_numpy()
            expected.append(average_precision_score(1 - frame.y, 1 - frame.p_success,
                                                    sample_weight=weights))
            legacy.append(average_precision_score(1 - frame.y, -frame.p_success,
                                                  sample_weight=weights))
        np.testing.assert_allclose(np.asarray(legacy) + correction, expected,
                                   rtol=0, atol=3e-16)
        np.testing.assert_allclose(original["pr_auc_risk"], expected, rtol=0, atol=3e-16)
        np.testing.assert_allclose(
            weighted_cluster_metric_draws_exact(frame, counts, order)["pr_auc_risk"],
            expected, rtol=0, atol=3e-16,
        )
        np.testing.assert_allclose(
            _weighted_cluster_average_precision_draws(frame, counts, order, batch_size=2),
            expected, rtol=0, atol=3e-16,
        )

    def test_random_cluster_weights_preserve_roc_and_risk_ap_definitions(self):
        rng = np.random.default_rng(71)
        frame = pd.DataFrame(dict(cluster_id=np.repeat(["C", "A", "B"], 20),
                                  y=np.tile([0, 1], 30),
                                  p_success=rng.choice([.0, .1, .4, .7, 1.0], 60)))
        order = np.array(["B", "C", "A"])
        counts = rng.multinomial(3, [1 / 3] * 3, size=17)
        actual = _weighted_cluster_metric_draws(frame, counts, order, batch_size=3)
        fast = _weighted_cluster_average_precision_draws(frame, counts, order, batch_size=4)
        for index, draw in enumerate(counts):
            weights = frame.cluster_id.map(dict(zip(order, draw))).to_numpy()
            self.assertAlmostEqual(actual["auc"][index], roc_auc_score(
                frame.y, frame.p_success, sample_weight=weights), places=14)
            expected = average_precision_score(1 - frame.y, 1 - frame.p_success,
                                               sample_weight=weights)
            self.assertAlmostEqual(actual["pr_auc_risk"][index], expected, places=14)
            self.assertAlmostEqual(fast[index], expected, places=14)

    def test_no_collapsed_thresholds_require_no_adjustment(self):
        frame = pd.DataFrame(dict(cluster_id=["A", "A", "B", "B"],
                                  y=[0, 1, 0, 1], p_success=[.1, .6, .3, .9]))
        counts = np.array([[1, 1], [2, 0], [0, 3]])
        self.assertEqual(merged_risk_thresholds(frame), 0)
        np.testing.assert_array_equal(
            average_precision_tie_correction(frame, counts, np.array(["A", "B"])),
            np.zeros(3),
        )

    def test_undefined_draw_policy_is_preserved(self):
        high = 0.30666666666666664
        low = float(np.nextafter(high, 0))
        frame = pd.DataFrame(dict(cluster_id=["A", "B"], y=[0, 1], p_success=[low, high]))
        counts = np.array([[0, 0], [1, 0], [0, 1]])
        corrected = weighted_cluster_metric_draws_exact(frame, counts, np.array(["A", "B"]))
        self.assertTrue(np.isnan(corrected["pr_auc_risk"]).all())
        self.assertTrue(np.isnan(_weighted_cluster_average_precision_draws(
            frame, counts, np.array(["A", "B"]))).all())


if __name__ == "__main__":
    unittest.main()
