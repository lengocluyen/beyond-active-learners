import unittest

import numpy as np
import pandas as pd

from scripts.aggregate_protocol_comparison import METRICS, SPECS, summarize, cell_estimates, AA, AV, VA, VV


class AggregateProtocolComparisonTests(unittest.TestCase):
    def test_canonical_named_draws_ignore_per_cell_row_order(self):
        rng = np.random.default_rng(23)
        frame = pd.DataFrame([
            dict(protocol=protocol, cluster_id=cluster, y=y, p_success=float(rng.uniform(.1, .9)))
            for protocol in (AA, AV, VA, VV) for cluster in ("C", "A", "B") for y in (0, 1)
        ])
        order = np.array(["A", "B", "C"])
        counts = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
        original = cell_estimates(frame, counts, order)
        shuffled = cell_estimates(frame.sample(frac=1, random_state=41), counts, order)
        for key in original:
            self.assertAlmostEqual(original[key]["point"], shuffled[key]["point"], places=14)
            np.testing.assert_allclose(original[key]["draws"], shuffled[key]["draws"], rtol=0, atol=1e-15)

    def test_observed_means_and_paired_vectors_keep_cell_identity(self):
        cells = {}
        for index, (point, draws) in enumerate(((.1, [.05, .2]), (.3, [.5, .2]))):
            cells[(2, str(index))] = {(metric, contrast): dict(point=point, draws=np.array(draws))
                                      for metric in METRICS for contrast in SPECS}
        rows, vectors = summarize(cells, "main")
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["cells"] == 2 for row in rows))
        self.assertAlmostEqual(rows[0]["point_mean"], .2)
        self.assertAlmostEqual(rows[0]["bootstrap_mean"], .2375)
        np.testing.assert_allclose(vectors[("auc", "joint_population_difference")][1], [.275, .2])

    def test_magnitude_difference_uses_aggregate_components(self):
        cells = {}
        for index, evaluation in enumerate((-.2, .6)):
            cells[(2, str(index))] = {(metric, contrast): dict(point=.1, draws=np.array([.1, .1]))
                                      for metric in METRICS for contrast in SPECS}
            cells[(2, str(index))][("auc", "evaluation_population_shapley")] = dict(
                point=evaluation, draws=np.array([evaluation, evaluation]))
        rows, _ = summarize(cells, "main")
        gap = next(row for row in rows if row["metric"] == "auc" and
                   row["contrast"] == "absolute_evaluation_minus_absolute_training")
        self.assertAlmostEqual(gap["point_mean"], .1)
        self.assertAlmostEqual(gap["ci_low"], .1)


if __name__ == "__main__":
    unittest.main()
