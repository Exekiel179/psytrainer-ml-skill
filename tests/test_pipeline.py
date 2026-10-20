import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline_train as runner


class PipelineTests(unittest.TestCase):
    def data(self, root):
        ids = [f"{i:03}" for i in range(80)]
        rng = np.random.default_rng(12)
        x = pd.DataFrame(rng.normal(size=(80, 3)), index=ids, columns=["a", "b", "c"])
        y = pd.DataFrame({"score": x.a * 2 - x.b + rng.normal(size=80) * .1}, index=ids)
        meta = pd.DataFrame({"subject": np.repeat(np.arange(20), 4),
                             "time": np.repeat(np.arange(40), 2)}, index=ids)
        for name, frame in (("x", x), ("y", y), ("meta", meta)):
            frame.to_csv(root / f"{name}.csv")
        args = runner.parser().parse_args(["train", "--features", str(root / "x.csv"),
            "--labels", str(root / "y.csv"), "--target", "score", "--task", "regression",
            "--output-dir", str(root / "out"), "--cv", "3", "--repeats", "2"])
        args.metric = "mae"
        return args, x, y.score, meta

    def test_group_and_time_partitions_never_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            args, x, y, meta = self.data(Path(temp))
            for mode, column in (("group", "subject"), ("time", "time")):
                args.split = mode
                args.gap = 1 if mode == "time" else 0
                m = meta[column]
                tr, te = runner.holdout_indices(x, y, m, args)
                self.assertFalse(set(m.iloc[tr]) & set(m.iloc[te]))
                if mode == "time":
                    self.assertLess(m.iloc[tr].max() + 1, m.iloc[te].min())
                for ft, fv in runner.cv_indices(x.iloc[tr], y.iloc[tr], m.iloc[tr], args):
                    mt, mv = m.iloc[tr].iloc[ft], m.iloc[tr].iloc[fv]
                    self.assertFalse(set(mt) & set(mv))
                    if mode == "time":
                        self.assertLess(mt.max() + 1, mv.min())

    def test_imputation_is_fold_local_and_saved_prediction_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.impute = "median"
            args.model = ["ModelLRRegressor"]
            x.loc["000", "a"] = np.nan
            x.to_csv(args.features)
            seen = []
            fit = SimpleImputer.fit

            def spy(instance, frame, target=None):
                seen.append(set(frame.index))
                return fit(instance, frame, target)

            with patch.object(runner, "estimator", return_value=LinearRegression()), \
                 patch.object(SimpleImputer, "fit", spy), patch("pipeline_report.generate_report"):
                result = runner.run(args)
            membership = pd.read_csv(root / "out/split-membership.csv", dtype={"sample_id": str})
            dev = set(membership.loc[membership.partition == "development", "sample_id"])
            test = set(membership.loc[membership.partition == "test", "sample_id"])
            self.assertEqual(len(seen), 4)
            self.assertEqual(seen[-1], dev)
            self.assertTrue(all(not rows & test and rows <= dev for rows in seen))
            self.assertTrue(all(len(rows) < len(dev) for rows in seen[:-1]))
            saved = joblib.load(result["model"])
            self.assertAlmostEqual(saved["pipeline"].named_steps["impute"].statistics_[0], x.loc[sorted(dev), "a"].median())
            x[["c", "a", "b"]].to_csv(root / "new.csv")
            pa = runner.parser().parse_args(["predict", "--features", str(root / "new.csv"),
                "--model", result["model"], "--output", str(root / "pred.csv")])
            runner.predict(pa)
            actual = pd.read_csv(root / "pred.csv").score
            np.testing.assert_allclose(actual, saved["pipeline"].predict(x))
            with self.assertRaises(FileExistsError):
                runner.predict(pa)

    def test_external_test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, meta = self.data(root)
            args.test_features, args.test_labels = args.features, args.labels
            with self.assertRaisesRegex(ValueError, "disjoint"):
                runner.run(args)
            x.index = ["test" + i for i in x.index]
            y.index = x.index
            meta.index = x.index
            x.to_csv(root / "test-x.csv")
            y.to_frame().to_csv(root / "test-y.csv")
            meta.to_csv(root / "test-meta.csv")
            args.test_features, args.test_labels = root / "test-x.csv", root / "test-y.csv"
            args.split, args.group_column = "group", "subject"
            args.metadata, args.test_metadata = root / "meta.csv", root / "test-meta.csv"
            with self.assertRaisesRegex(ValueError, "shares groups"):
                runner.run(args)
            args.split, args.group_column, args.time_column = "time", None, "time"
            with self.assertRaisesRegex(ValueError, "strictly after"):
                runner.run(args)

    def test_r2_selects_higher_scores_and_test_labels_cannot_change_selection(self):
        from sklearn.dummy import DummyRegressor
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.metric, args.model = "r2", ["ModelRandomForestRegressor", "ModelLRRegressor"]
            x.index = ["test" + i for i in x.index]
            y.index = x.index
            args.test_features, args.test_labels = root / "tx.csv", root / "ty.csv"
            x.to_csv(args.test_features)
            y.to_frame().to_csv(args.test_labels)
            factory = lambda tag, *_: DummyRegressor() if tag == "ModelRandomForestRegressor" else LinearRegression()
            with patch.object(runner, "estimator", side_effect=factory), patch("pipeline_report.generate_report"):
                first = runner.run(args)
                args.output_dir = root / "out2"
                (y * -100).to_frame().to_csv(args.test_labels)
                second = runner.run(args)
            self.assertEqual(first["selected_model"], "ModelLRRegressor")
            self.assertEqual(second["selected_model"], "ModelLRRegressor")
            one = json.loads(Path(first["summary"]).read_text())
            two = json.loads(Path(second["summary"]).read_text())
            self.assertEqual(one["comparison"], two["comparison"])
            self.assertNotEqual(one["test_metrics"], two["test_metrics"])

    def test_explicit_model_parameters_survive_training_and_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _, _ = self.data(root)
            args.model = ["ModelRandomForestRegressor"]
            args.model_params = root / "params.json"
            args.model_params.write_text(json.dumps({args.model[0]: {"n_estimators": 7, "max_depth": 2}}))
            with patch("pipeline_report.generate_report"):
                result = runner.run(args)
            saved = joblib.load(result["model"])["pipeline"].named_steps["model"]
            self.assertEqual(saved.n_estimators, 7)
            self.assertEqual(saved.max_depth, 2)
            summary = json.loads(Path(result["summary"]).read_text())
            self.assertIn("model_params", summary["inputs"])
            self.assertEqual(summary["model_parameters"][args.model[0]]["max_depth"], 2)

    def test_missing_class_and_invalid_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, x, y, meta = self.data(Path(temp))
            args.split, args.task, args.metric = "time", "classification", "accuracy"
            classes = pd.Series(["a"] * 40 + ["b"] * 40, index=x.index)
            with self.assertRaisesRegex(ValueError, "lacks a class"):
                runner.cv_indices(x, classes, meta.time, args)
            args.repeats = 1
            with self.assertRaisesRegex(ValueError, "repeats"):
                runner.run(args)

    def test_class_order_and_permutation_test_helpers(self):
        self.assertEqual(runner.class_order(["case", "control", "case"]), ["case", "control"])
        self.assertEqual(runner.class_order(["case", "control"], "case"), ["control", "case"])
        with self.assertRaisesRegex(ValueError, "not a class"):
            runner.class_order(["a", "b"], "c")
        with self.assertRaisesRegex(ValueError, "binary"):
            runner.class_order(["a", "b", "c"], "a")
        labels = pd.Series(np.r_[np.zeros(50), np.ones(50)])
        # Score equals the fraction of labels left in their original position: high only when unshuffled.
        result = runner.permutation_test(lambda y: y, lambda y: float(np.mean(y.to_numpy() == labels.to_numpy())),
                                         labels, 199, 0)
        self.assertEqual(result["exceed"], 0)
        self.assertAlmostEqual(result["p_value"], 1 / 200)
        self.assertEqual(result["valid"], 199)

    def test_positive_class_changes_reading_not_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, meta = self.data(root)
            rng = np.random.default_rng(3)
            classes = pd.DataFrame({"group": np.where(x.a + rng.normal(size=len(x)) * .5 > 0, "case", "control")}, index=x.index)
            classes.to_csv(root / "c.csv")
            summaries = {}
            for name, positive in (("default", None), ("case", "case")):
                argv = ["train", "--features", str(root / "x.csv"), "--labels", str(root / "c.csv"), "--target", "group",
                        "--task", "classification", "--output-dir", str(root / name), "--cv", "3", "--repeats", "2",
                        "--bootstrap", "0", "--model", "ModelLRClassifier", "--model", "ModelRandomForestClassifier"]
                if positive:
                    argv += ["--positive-class", positive, "--permutations", "99"]
                with patch("pipeline_report.generate_report"):
                    runner.run(runner.parser().parse_args(argv))
                summaries[name] = json.loads((root / name / "summary.json").read_text())
            a, b = summaries["default"], summaries["case"]
            self.assertEqual(a["positive_class"], "control")
            self.assertEqual(b["positive_class"], "case")
            self.assertEqual(b["classes"], ["control", "case"])
            self.assertEqual(a["comparison"], b["comparison"])
            for key in ("accuracy", "balanced_accuracy", "roc_auc"):
                self.assertAlmostEqual(a["test_metrics"][key], b["test_metrics"][key])
            test = b["permutation_test"]
            self.assertEqual(test["valid"], 99)
            self.assertLess(test["p_value"], .05)
            self.assertTrue((root / "case" / "permutation-null.csv").is_file())
            self.assertIsNone(a["permutation_test"])
            pred = pd.read_csv(root / "case" / "test-predictions.csv")
            case_rate = pred.loc[pred.observed == "case", "positive_score"].mean()
            control_rate = pred.loc[pred.observed == "control", "positive_score"].mean()
            self.assertGreater(case_rate, control_rate)


if __name__ == "__main__":
    unittest.main()
