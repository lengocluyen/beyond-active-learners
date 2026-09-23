from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.cohort_exchange import (
    CROSS_PROTOCOL,
    CROSS_PROTOCOL_VA,
    KEY,
    METRICS,
    _prediction_protocols_for_dataset,
    _train_evaluation_specs,
    _weighted_cluster_metric_draws,
    average_repeated_predictions,
    budget_and_coverage_table,
    cluster_bootstrap_decomposition,
    cohort_membership,
    composition_table,
    decision_overlap_table,
    metric_values,
    reference_baseline_manifest,
    select_at_budget,
)
from src.full_evaluation import REFERENCE_BASELINES, _model


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, "A", "P1", 0, -20, 10],   # outcome realised before week 2
            [2, "A", "P1", 0, -20, np.nan],  # eligible and active
            [3, "A", "P1", 0, -20, np.nan],  # eligible and silent
            [4, "A", "P1", 1, 30, np.nan],   # not registered at week 2
        ],
        columns=KEY + ["label", "date_registration", "date_unregistration"],
    ).assign(risk=lambda x: 1 - x["label"])


class CohortMembershipTests(unittest.TestCase):
    def test_cutoff_membership_separates_known_and_silent(self) -> None:
        events = pd.DataFrame(
            [
                [1, "A", "P1", 1],
                [2, "A", "P1", 1],
            ],
            columns=KEY + ["week_index"],
        )
        result = cohort_membership("oulab", 2, roster=_roster(), events=events)
        classes = result.set_index("id_student")["membership_class"].to_dict()
        self.assertEqual(classes[1], "outcome_realized_active")
        self.assertEqual(classes[2], "eligible_active")
        self.assertEqual(classes[3], "eligible_silent")
        self.assertEqual(classes[4], "not_yet_registered")
        self.assertEqual(int(result["activity_conditioned"].sum()), 2)
        self.assertEqual(int(result["cutoff_valid"].sum()), 2)

    def test_composition_indices_detect_exchange(self) -> None:
        events = pd.DataFrame(
            [[1, "A", "P1", 1], [2, "A", "P1", 1]],
            columns=KEY + ["week_index"],
        )
        membership = cohort_membership("oulab", 2, roster=_roster(), events=events)
        detailed = composition_table(membership)
        indices = detailed.attrs["indices"].iloc[0]
        self.assertEqual(indices["known_outcomes_in_activity"], 1)
        self.assertEqual(indices["eligible_silent_excluded"], 1)
        self.assertAlmostEqual(indices["cohort_jaccard"], 1 / 3)


class PredictionAggregationTests(unittest.TestCase):
    def test_repeats_are_averaged_per_learner(self) -> None:
        base = {
            "dataset": "oulab",
            "week": 2,
            "protocol": "cutoff_valid",
            "model": "flat_hgb",
            "id_student": 1,
            "code_module": "A",
            "code_presentation": "P1",
            "y": 0,
            "risk": 1,
            "has_activity": True,
            "no_activity": 0,
            "outcome_realized": False,
            "cutoff_valid": True,
            "membership_class": "eligible_active",
            "cluster_id": "A::P1",
            "cluster_unit": "presentation",
        }
        predictions = pd.DataFrame(
            [
                {**base, "repeat": 1, "p_success": 0.2, "risk_score": 0.8},
                {**base, "repeat": 2, "p_success": 0.4, "risk_score": 0.6},
            ]
        )
        averaged = average_repeated_predictions(predictions)
        self.assertEqual(len(averaged), 1)
        self.assertAlmostEqual(averaged.loc[0, "p_success"], 0.3)
        self.assertEqual(averaged.loc[0, "n_repeats"], 2)
        self.assertEqual(averaged.loc[0, "train_protocol"], "cutoff_valid")
        self.assertEqual(averaged.loc[0, "eval_protocol"], "cutoff_valid")

    def test_budget_selection_respects_cluster_capacity(self) -> None:
        scored = pd.DataFrame(
            {
                "cluster_id": ["A", "A", "B", "B"],
                "risk_score": [0.9, 0.1, 0.8, 0.2],
            }
        )
        selected = select_at_budget(scored, {"A": 1, "B": 1})
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["risk_score"]), {0.9, 0.8})

    def test_budget_metrics_treat_known_outcomes_as_wasted_slots(self) -> None:
        common = {
            "dataset": "oulab",
            "week": 2,
            "model": "flat_hgb",
            "code_module": "A",
            "code_presentation": "P1",
            "cluster_id": "A::P1",
        }
        predictions = pd.DataFrame(
            [
                {**common, "protocol": "cutoff_valid", "id_student": 1,
                 "risk": 1, "risk_score": 0.9},
                {**common, "protocol": "cutoff_valid", "id_student": 2,
                 "risk": 0, "risk_score": 0.2},
                {**common, "protocol": "cutoff_valid", "id_student": 3,
                 "risk": 1, "risk_score": 0.8},
                {**common, "protocol": "activity_conditioned", "id_student": 1,
                 "risk": 1, "risk_score": 0.9},
                {**common, "protocol": "activity_conditioned", "id_student": 4,
                 "risk": 1, "risk_score": 0.95},
            ]
        )
        table, _ = budget_and_coverage_table(
            predictions, pd.DataFrame(), budgets=(0.5,)
        )
        row = table[table["protocol"] == "activity_conditioned"].iloc[0]
        self.assertEqual(row["n_invalid_flagged"], 1)
        self.assertAlmostEqual(row["wasted_budget_rate"], 0.5)
        self.assertAlmostEqual(row["actionable_precision"], 0.5)
        self.assertAlmostEqual(row["coverage_adjusted_recall"], 0.5)
        self.assertAlmostEqual(row["candidate_coverage"], 1 / 3)

    def test_decision_overlap_includes_pure_evaluation_population_contrast(self) -> None:
        common = {
            "dataset": "oulab",
            "week": 2,
            "model": "flat_hgb",
            "budget": 0.05,
            "code_module": "A",
            "code_presentation": "P1",
        }
        selections = pd.DataFrame(
            [
                {**common, "protocol": "cutoff_valid", "id_student": 1},
                {**common, "protocol": "cutoff_valid", "id_student": 2},
                {**common, "protocol": "activity_conditioned", "id_student": 1},
                {**common, "protocol": "activity_conditioned", "id_student": 3},
                {**common, "protocol": CROSS_PROTOCOL, "id_student": 1},
                {**common, "protocol": CROSS_PROTOCOL, "id_student": 2},
            ]
        )
        overlap = decision_overlap_table(selections)
        row = overlap[
            overlap["comparison_type"].eq("evaluation_population")
        ].iloc[0]
        self.assertAlmostEqual(row["jaccard"], 1 / 3)
        self.assertEqual(row["reference_protocol"], CROSS_PROTOCOL)


class ClusterBootstrapTests(unittest.TestCase):
    def test_weighted_draws_equal_materialized_cluster_resamples(self) -> None:
        predictions = pd.DataFrame(
            {
                "cluster_id": ["A", "A", "B", "B", "C", "C"],
                "y": [1, 0, 1, 0, 1, 0],
                "p_success": [0.8, 0.4, 0.6, 0.2, 0.5, 0.5],
            }
        )
        clusters = np.array(["A", "B", "C"])
        counts = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 1]])
        actual = _weighted_cluster_metric_draws(predictions, counts, clusters)

        for draw, frequencies in enumerate(counts):
            chunks = []
            for cluster, frequency in zip(clusters, frequencies):
                chunks.extend(
                    [predictions[predictions["cluster_id"] == cluster]]
                    * int(frequency)
                )
            expected = metric_values(pd.concat(chunks, ignore_index=True))
            for metric in METRICS:
                self.assertAlmostEqual(actual[metric][draw], expected[metric])

    def test_cross_protocol_decomposition_is_exact(self) -> None:
        rows = []
        probabilities = {
            "activity_conditioned": [0.9, 0.3, 0.8, 0.2],
            CROSS_PROTOCOL: [0.8, 0.4, 0.7, 0.3],
            CROSS_PROTOCOL_VA: [0.6, 0.7, 0.8, 0.1],
            "cutoff_valid": [0.7, 0.5, 0.6, 0.4],
        }
        for protocol, scores in probabilities.items():
            for index, (y, score) in enumerate(zip([1, 0, 1, 0], scores)):
                rows.append(
                    {
                        "dataset": "oulab",
                        "week": 2,
                        "model": "flat_hgb",
                        "protocol": protocol,
                        "cluster_id": "A" if index < 2 else "B",
                        "cluster_unit": "presentation",
                        "y": y,
                        "p_success": score,
                    }
                )
        result = cluster_bootstrap_decomposition(
            pd.DataFrame(rows), n_boot=20, seed=7
        )
        self.assertEqual(len(result), 7 * len(METRICS))
        self.assertTrue(np.allclose(result["identity_residual"], 0.0))
        points = {
            protocol: metric_values(pd.DataFrame({
                "y": [1, 0, 1, 0], "p_success": scores,
            }))
            for protocol, scores in probabilities.items()
        }
        for metric, group in result.groupby("metric"):
            deltas = group.set_index("contrast")["delta"]
            self.assertAlmostEqual(
                deltas["joint_population_difference"],
                deltas["evaluation_population_component"]
                + deltas["training_population_component"],
                msg=metric,
            )
            evaluation = 0.5 * (
                points["activity_conditioned"][metric] - points[CROSS_PROTOCOL][metric]
                + points[CROSS_PROTOCOL_VA][metric] - points["cutoff_valid"][metric]
            )
            training = 0.5 * (
                points["activity_conditioned"][metric] - points[CROSS_PROTOCOL_VA][metric]
                + points[CROSS_PROTOCOL][metric] - points["cutoff_valid"][metric]
            )
            self.assertAlmostEqual(deltas["evaluation_population_shapley"], evaluation, msg=metric)
            self.assertAlmostEqual(deltas["training_population_shapley"], training, msg=metric)
            self.assertAlmostEqual(deltas["joint_population_difference"], evaluation + training, msg=metric)
        # A hand-computed, nontrivial example guards against swapping the two
        # equal-weight components: Brier cell values are .045, .095, .175, .165.
        brier = result.loc[result.metric.eq("brier")].set_index("contrast")["delta"]
        self.assertAlmostEqual(brier["evaluation_population_shapley"], -0.02)
        self.assertAlmostEqual(brier["training_population_shapley"], -0.10)


class TrainEvaluationProtocolTests(unittest.TestCase):
    def test_oulad_adds_cross_evaluation_without_duplicate_training_cell(self) -> None:
        protocols = _prediction_protocols_for_dataset("oulab")
        specs = _train_evaluation_specs("oulab")
        self.assertIn(CROSS_PROTOCOL, protocols)
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            specs["activity_conditioned"],
            (
                ("activity_conditioned", "activity_conditioned"),
                (CROSS_PROTOCOL, "cutoff_valid"),
            ),
        )

    def test_kdd_remains_two_identifiable_cells(self) -> None:
        self.assertEqual(
            _prediction_protocols_for_dataset("kdd"),
            ("activity_conditioned", "static_full"),
        )


class ReferenceBaselineTests(unittest.TestCase):
    def test_reference_suite_has_traceable_manifest(self) -> None:
        models = REFERENCE_BASELINES["oulab"]
        manifest = reference_baseline_manifest(models)
        self.assertEqual(set(manifest["model"]), set(models))
        self.assertTrue(manifest["doi"].str.startswith("10.").all())
        self.assertTrue(manifest["scope"].str.contains("adaptation|transplant").all())

    def test_exact_knn_is_not_in_large_kdd_default_suite(self) -> None:
        self.assertNotIn("base_knn", REFERENCE_BASELINES["kdd"])
        self.assertIn("base_knn", REFERENCE_BASELINES["oulab"])

    def test_missing_paper_variants_have_distinct_estimators(self) -> None:
        rf_gini = _model("base_rf_gini", 42)
        rf_entropy = _model("base_rf_entropy", 42)
        dffnn = _model("base_dffnn", 42)
        self.assertEqual(rf_gini.criterion, "gini")
        self.assertEqual(rf_entropy.criterion, "entropy")
        self.assertEqual(
            dffnn.named_steps["model"].hidden_layer_sizes, (128, 64, 32)
        )


if __name__ == "__main__":
    unittest.main()
