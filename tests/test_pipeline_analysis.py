import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pipeline_analysis import (bootstrap_indices, calibration_table, feature_shift,
                               metric_values, regression_bins, threshold_table, uncertainty)
from pipeline_train import evaluate


class DiagnosticsTests(unittest.TestCase):
    def regression(self):
        observed = np.arange(30, dtype=float)
        frame = pd.DataFrame({"sample_id": [f"{i:03}" for i in range(30)],
                              "observed": observed, "predicted": observed - 1})
        summary = {"task": "regression", "classes": [], "split": "random", "seed": 4,
                   "bootstrap_repeats": 100}
        membership = pd.DataFrame({"sample_id": frame.sample_id, "split_value": np.repeat(np.arange(6), 5)})
        return frame, summary, membership

    def test_paired_bootstrap_known_gain_and_reproducibility(self):
        pred, summary, member = self.regression()
        baseline = pred.assign(predicted=pred.observed - 3)
        table, _ = uncertainty(pred, baseline, summary, member)
        row = table.set_index("metric").loc["mae"]
        self.assertEqual(row.estimate, 1)
        self.assertEqual(row.gain, 2)
        self.assertEqual(row.gain_lower, 2)
        self.assertEqual(row.gain_upper, 2)
        pd.testing.assert_frame_equal(table, uncertainty(pred, baseline, summary, member)[0])

    def test_clusters_preserve_entire_groups_and_time_omits_intervals(self):
        pred, summary, member = self.regression()
        groups = member.split_value.to_numpy()
        ix = bootstrap_indices(len(pred), np.random.default_rng(5), groups)
        counts = np.bincount(ix, minlength=len(pred))
        for unit in np.unique(groups):
            self.assertEqual(len(set(counts[groups == unit])), 1)
        summary["split"] = "group"
        table, info = uncertainty(pred, pred, summary, member)
        self.assertEqual(info["units"], 6)
        self.assertTrue(table.lower.notna().all())
        summary["split"] = "time"
        table, info = uncertainty(pred, pred, summary, member)
        self.assertTrue(table.lower.isna().all())
        self.assertIn("Time dependence", info["omitted_reason"])

    def test_missing_class_has_no_full_balanced_accuracy_or_auc(self):
        pred = pd.DataFrame({"observed": ["a", "a"], "predicted": ["a", "a"], "positive_score": [.1, .2]})
        values = metric_values(pred, "classification", ["a", "b"])
        self.assertIsNone(values["balanced_accuracy"])
        self.assertIsNone(values["roc_auc"])

    def test_single_class_auc_primary_report_analysis_does_not_fail(self):
        from pipeline_analysis import analyze
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pred = pd.DataFrame({"sample_id": ["001", "002"], "observed": ["NA", "NA"],
                "predicted": ["NA", "NA"], "positive_score": [.1, .2], "positive_probability": [.1, .2]})
            pred.to_csv(root / "test-predictions.csv", index=False)
            pred.to_csv(root / "baseline-predictions.csv", index=False)
            pd.DataFrame({"sample_id": pred.sample_id, "partition": "test", "split_value": ""}).to_csv(root / "split-membership.csv", index=False)
            pd.DataFrame({"model": ["lr"], "score": [.8]}).to_csv(root / "cv-scores.csv", index=False)
            summary = {"task": "classification", "classes": ["NA", "positive"], "split": "random",
                       "seed": 1, "bootstrap_repeats": 0, "metric": "roc_auc"}
            analyze(root, summary)
            table = pd.read_csv(root / "metric-intervals.csv").set_index("metric")
            self.assertTrue(pd.isna(table.loc["roc_auc", "estimate"]))
            self.assertEqual(json.loads((root / "analysis.json").read_text())["test_n"], 2)
            errors = pd.read_csv(root / "error-cases.csv", keep_default_na=False)
            self.assertEqual(list(errors.observed), ["NA", "NA"])

    def test_calibration_and_threshold_values(self):
        pred = pd.DataFrame({"observed": ["n", "p", "p", "n"], "predicted": ["n", "n", "p", "p"],
                             "positive_score": [0, .2, .8, 1], "positive_probability": [0, .2, .8, 1]})
        cal = calibration_table(pred, "p")
        self.assertEqual(cal.n.sum(), 4)
        self.assertEqual(cal.iloc[-1].bin, 9)
        self.assertEqual(cal.iloc[-1].observed_fraction, 0)
        row = threshold_table(pred, "p").set_index("threshold").loc[.5]
        self.assertEqual(row.tp, 1)
        self.assertEqual(row.fp, 1)
        self.assertEqual(row.sensitivity, .5)
        self.assertAlmostEqual(metric_values(pred, "classification", ["n", "p"])["brier"], .42)

    def test_decision_scores_in_unit_interval_are_not_probabilities(self):
        x = pd.DataFrame({"x": [-2, -1, 1, 2]})
        y = pd.Series(["n", "n", "p", "p"])
        _, scores = evaluate(LinearSVC().fit(x, y), x, y, "classification")
        self.assertIn("positive_score", scores)
        self.assertNotIn("positive_probability", scores)
        _, probabilities = evaluate(LogisticRegression().fit(x, y), x, y, "classification")
        self.assertIn("positive_probability", probabilities)

    def test_constant_bins_and_undefined_feature_scale(self):
        pred = pd.DataFrame({"observed": [0, 1, 2], "predicted": [1, 1, 1]})
        bins = regression_bins(pred)
        self.assertEqual(len(bins), 1)
        self.assertEqual(bins.n.sum(), 3)
        self.assertAlmostEqual(bins.mae.iloc[0], 2/3)
        a = pd.DataFrame({"constant": [1, 1], "missing": [np.nan, np.nan], "signal": [0, 2]})
        b = pd.DataFrame({"constant": [2, 2], "missing": [0, 1], "signal": [1, 3]})
        shift = feature_shift(a, b).set_index("feature")
        self.assertTrue(pd.isna(shift.loc["constant", "standardized_mean_shift"]))
        self.assertEqual(shift.loc["signal", "standardized_mean_shift"], 1)
        self.assertEqual(shift.loc["missing", "development_missing"], 1)


if __name__ == "__main__":
    unittest.main()
