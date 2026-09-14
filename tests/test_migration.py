"""Behavioral coverage for the independently migrated wheel capabilities."""

import json
import ast
import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline_train as runner
from pipeline_options import SAMPLERS, DOMAIN_MODELS, FThresholdSelector, search_candidates, make_sampler, pearson_score
import test_pipeline


class MigrationTests(unittest.TestCase):
    def data(self, root):
        args, x, y, meta = test_pipeline.PipelineTests().data(root)
        args.bootstrap = 0
        args.model = ["ModelLRRegressor"]
        return args, x, y, meta

    def test_f_threshold_includes_last_column_and_rejects_empty_selection(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(200, 3))
        y = 8 * x[:, -1] + rng.normal(size=200)
        selector = FThresholdSelector(f_regression, threshold=20).fit(x, y)
        self.assertEqual(selector.get_support().tolist(), [False, False, True])
        np.testing.assert_allclose(selector.transform(x).ravel(), x[:, -1])
        selector.scores_ = np.array([np.nan, 0, np.inf])
        self.assertEqual(selector.get_support().tolist(), [False, False, True])
        with self.assertRaisesRegex(ValueError, "no features"):
            FThresholdSelector(f_regression, threshold=1e100).fit_transform(x, y)

    def test_domain_presets_match_wheel_interfaces_and_train_without_vendor(self):
        root = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(root / "vendor/PsyTrainer-0.2.0-cp314-none-any.whl") as wheel:
            for domain, tags in DOMAIN_MODELS.items():
                tree = ast.parse(wheel.read(f"ccpl_training_models/apps/{domain}/interface.py"))
                original = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                            and isinstance(node.value, str) and node.value.startswith("Model")}
                self.assertEqual(set(tags), original)
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = self.data(Path(temp))
            args.model = None
            with patch.dict(sys.modules, {"ccpl_training_models": None}), patch("pipeline_report.generate_report"):
                for domain, tags in DOMAIN_MODELS.items():
                    args.preset, args.output_dir = domain, Path(temp) / domain
                    result = runner.run(args)
                    self.assertEqual(result["failures"], [])
                    summary = json.loads(Path(result["summary"]).read_text())
                    self.assertEqual({row["model"] for row in summary["comparison"]}, set(tags))

    def test_combination_search_and_deprecated_scoring_boundaries(self):
        self.assertTrue(np.isnan(pearson_score([1, 2, 3], [1, 1, 1])))
        self.assertNotIn("pearson_p", runner.SCORERS["regression"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.task, args.metric, args.model = "classification", "recall", ["ModelLRClassifier"]
            pd.DataFrame({"score": np.where(y > y.median(), "case", "control")}, index=x.index).to_csv(args.labels)
            args.search, args.search_space = "grid", root / "space.json"
            args.search_space.write_text(json.dumps({args.model[0]: [
                {"preprocess__selection": ["kbest"], "preprocess__select_k": [1, 2],
                 "preprocess__resample": ["none", "random-over"]},
                {"preprocess__selection": ["forward"], "preprocess__select_k": [1]}]}))
            with patch("pipeline_report.generate_report"):
                result = runner.run(args)
            summary = json.loads(Path(result["summary"]).read_text())
            self.assertEqual(summary["search"]["evaluated_candidates"], 5)
            self.assertEqual(summary["search"]["failed_candidates"], 0)
            fitted = joblib.load(result["model"])["pipeline"]
            self.assertEqual(summary["preprocessing"]["select_k"], fitted.named_steps["select"].get_support().sum())

    def test_minmax_fits_development_only_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.scaler = "minmax"
            args.selection = "f-threshold"
            args.f_threshold = 0
            with patch("pipeline_report.generate_report"):
                result = runner.run(args)
            fitted = joblib.load(result["model"])["pipeline"]
            membership = pd.read_csv(root / "out/split-membership.csv", dtype={"sample_id": str})
            dev = membership.loc[membership.partition == "development", "sample_id"]
            np.testing.assert_allclose(fitted.named_steps["scale"].data_max_, x.loc[dev].max())
            predicted = fitted.predict(x)
            joblib.dump(fitted, root / "copy.joblib")
            np.testing.assert_allclose(predicted, joblib.load(root / "copy.joblib").predict(x))

    def test_all_seven_samplers_fit_predict_and_do_not_resample_prediction(self):
        x, y = make_classification(n_samples=180, n_features=4, n_informative=3,
            n_redundant=0, weights=[.75, .25], random_state=8)
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = self.data(Path(temp))
            args.task, args.metric = "classification", "accuracy"
            args.sampler_parameters = {}
            for name in SAMPLERS:
                with self.subTest(sampler=name):
                    args.resample = name
                    pipe = runner.make_pipeline(LogisticRegression(), args)
                    sampler = type(pipe.named_steps["resample"])
                    original = sampler.fit_resample
                    seen = []

                    def spy(instance, features, labels, **kwargs):
                        seen.append(len(features))
                        return original(instance, features, labels, **kwargs)

                    with patch.object(sampler, "fit_resample", spy):
                        fitted = clone(pipe).fit(x[:140], y[:140])
                        prediction = fitted.predict(x[140:])
                    self.assertEqual(seen, [140])
                    self.assertEqual(len(prediction), 40)
                    path = Path(temp) / "sampler.joblib"
                    joblib.dump(fitted, path)
                    np.testing.assert_array_equal(prediction, joblib.load(path).predict(x[140:]))
            with self.assertRaises(ValueError):
                make_sampler("smote", 42, {"k_neighbors": 100}).fit_resample(x[:20], y[:20])

    def test_search_fold_locality_invalid_candidates_and_holdout_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.search, args.impute = "grid", "median"
            args.search_space = root / "search.json"
            args.search_space.write_text(json.dumps({args.model[0]: [
                {"fit_intercept": [True, False], "preprocess__scaler": ["standard", "minmax"]},
                {"nonexistent_parameter": [1]}]}))
            tx, ty = x.copy(), y.copy()
            tx.index = ty.index = ["test" + key for key in x.index]
            args.test_features, args.test_labels = root / "tx.csv", root / "ty.csv"
            tx.to_csv(args.test_features)
            ty.to_frame().to_csv(args.test_labels)
            seen = []
            original = SimpleImputer.fit

            def spy(instance, features, labels=None):
                seen.append(set(features.index))
                return original(instance, features, labels)

            with patch.object(SimpleImputer, "fit", spy), patch("pipeline_report.generate_report"):
                first = runner.run(args)
                args.output_dir = root / "second"
                (ty * -100).to_frame().to_csv(args.test_labels)
                second = runner.run(args)
            a, b = [json.loads(Path(result["summary"]).read_text()) for result in (first, second)]
            self.assertEqual(a["search"], b["search"])
            self.assertEqual(a["comparison"], b["comparison"])
            self.assertNotEqual(a["test_metrics"], b["test_metrics"])
            self.assertEqual(a["search"]["failed_candidates"], 1)
            self.assertEqual(len(seen), 2 * (4 * args.cv + 1))
            self.assertTrue(all(ids <= set(x.index) for ids in seen))
            self.assertIn("validation_mse", pd.read_csv(root / "out/cv-scores.csv"))

    def test_forward_selection_inner_splits_and_imputation_are_local(self):
        with tempfile.TemporaryDirectory() as temp:
            args, x, y, meta = self.data(Path(temp))
            args.selection, args.select_k, args.impute = "forward", 1, "median"
            args.sampler_parameters = {}
            x.iloc[0, 0] = np.nan
            for split, column in (("random", None), ("group", "subject"), ("time", "time")):
                with self.subTest(split=split):
                    args.split = split
                    m = meta[column] if column else None
                    inner = runner.cv_indices(x, y, m, args)
                    fitted = runner.configured_pipeline(LinearRegression(), args, {}, x, y, m)
                    actual = fitted.named_steps["select"].cv
                    for (tr, va), (atr, ava) in zip(inner, actual):
                        np.testing.assert_array_equal(tr, atr)
                        np.testing.assert_array_equal(va, ava)
                        if m is not None:
                            self.assertFalse(set(m.iloc[tr]) & set(m.iloc[va]))
                            if split == "time":
                                self.assertLess(m.iloc[tr].max(), m.iloc[va].min())
                    seen = []
                    original = SimpleImputer.fit

                    def spy(instance, features, labels=None):
                        seen.append(len(features))
                        return original(instance, features, labels)

                    with patch.object(SimpleImputer, "fit", spy):
                        fitted.fit(x, y)
                    self.assertEqual(seen[-1], len(x))
                    self.assertTrue(all(n < len(x) for n in seen[:-1]))
                    self.assertEqual(fitted.named_steps["select"].get_support().sum(), 1)
                    joblib.dump(fitted, Path(temp) / "forward.joblib")
                    np.testing.assert_allclose(fitted.predict(x), joblib.load(Path(temp) / "forward.joblib").predict(x))

    def test_resume_skips_completed_folds_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            original = LinearRegression.fit
            calls = []

            def interrupted(instance, features, labels, **kwargs):
                calls.append(len(features))
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return original(instance, features, labels, **kwargs)

            with patch.object(LinearRegression, "fit", interrupted), patch("pipeline_report.generate_report"):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(args)
            self.assertEqual(len(list((root / "out/checkpoints").glob("*.json"))), 1)
            args.resume = True
            calls.clear()

            def spy(instance, features, labels, **kwargs):
                calls.append(len(features))
                return original(instance, features, labels, **kwargs)

            with patch.object(LinearRegression, "fit", spy), patch("pipeline_report.generate_report"):
                result = runner.run(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(calls), args.cv)  # Remaining folds plus final refit.
            x.iloc[0, 0] += 1
            x.to_csv(args.features)
            with self.assertRaisesRegex(ValueError, "identical"):
                runner.run(args)

    def test_all_model_compact_grids_fit_and_random_budget_is_reproducible(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(40, 3))
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = self.data(Path(temp))
            args.search, args.max_candidates = "random", 2
            with patch.dict(sys.modules, {"ccpl_training_models": None}):
                for task, tags in runner.MODELS.items():
                    y = (x[:, 0] > 0).astype(int) if task == "classification" else x[:, 0] * 2
                    for tag in tags:
                        with self.subTest(tag=tag):
                            candidates = search_candidates(tag, args, {})
                            self.assertEqual(candidates, search_candidates(tag, args, {}))
                            self.assertLessEqual(len(candidates), 2)
                            model = runner.create_estimator(tag, task, 42)
                            model.set_params(**{k.removeprefix("model__"): v for k, v in candidates[0].items()})
                            if "CatBoost" in tag:
                                model.set_params(allow_writing_files=False, verbose=False)
                            with contextlib.redirect_stdout(io.StringIO()):
                                model.fit(x, y)
                            self.assertTrue(np.isfinite(model.predict(x)).all())
            args.search, args.max_candidates = "grid", 1
            with self.assertRaisesRegex(ValueError, "exceed"):
                search_candidates("ModelLRRegressor", args, {})
            with self.assertRaisesRegex(ValueError, "cannot override"):
                search_candidates("ModelLRRegressor", args, {"ModelLRRegressor": {"select__cv": [2]}})

    def test_xgboost_string_labels_and_saved_candidates_predict_original_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, x, y, _ = self.data(root)
            args.task, args.metric = "classification", "precision"
            args.model, args.save_candidates = ["ModelXGBoostClassifier", "ModelLRClassifier"], True
            pd.DataFrame({"score": np.where(y > y.median(), "case", "control")}, index=x.index).to_csv(args.labels)
            with patch("pipeline_report.generate_report"):
                result = runner.run(args)
            self.assertEqual(result["failures"], [])
            for path in (root / "out/candidates").glob("*.joblib"):
                pa = runner.parser().parse_args(["predict", "--features", str(args.features), "--model", str(path),
                                                "--output", str(root / f"{path.stem}.csv"), "--probabilities"])
                runner.predict(pa)
                self.assertEqual(set(pd.read_csv(pa.output).score), {"case", "control"})
                np.testing.assert_allclose(pd.read_csv(pa.output).filter(like="probability::").sum(axis=1), 1)
            self.assertEqual(set(pd.read_csv(root / "out/test-predictions.csv").observed), {"case", "control"})


if __name__ == "__main__":
    unittest.main()
