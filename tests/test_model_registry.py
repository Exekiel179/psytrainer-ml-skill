import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_registry as registry
import ml


class RegistryTests(unittest.TestCase):
    def test_all_models_fit_predict_and_round_trip_without_vendor(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(32, 3))
        with patch.dict(sys.modules, {"ccpl_training_models": None}), tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(registry.MODELS["classification"]), 9)
            self.assertEqual(len(registry.MODELS["regression"]), 12)
            for task, models in registry.MODELS.items():
                y = (x[:, 0] > 0).astype(int) if task == "classification" else x[:, 0] * 2 + x[:, 1]
                for tag in models:
                    with self.subTest(tag=tag), contextlib.redirect_stdout(io.StringIO()):
                        model = clone(registry.create_estimator(tag, task, 42))
                        if tag == "ModelCatBoostRegressor":
                            model.set_params(train_dir=temp)
                        model.fit(x, y)
                        predicted = model.predict(x)
                        self.assertTrue(np.isfinite(predicted).all())
                        path = Path(temp) / "model.joblib"
                        joblib.dump(model, path)
                        np.testing.assert_allclose(predicted, joblib.load(path).predict(x))

    def test_parameters_and_validation(self):
        model = registry.create_estimator("ModelRandomForestRegressor", "regression", 42,
                                          {"max_depth": 3, "n_estimators": 7, "random_state": 10})
        self.assertEqual(model.max_depth, 3)
        self.assertEqual(model.n_estimators, 7)
        self.assertEqual(model.random_state, 10)
        self.assertEqual(model.n_jobs, 1)
        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            registry.create_estimator("ModelLRRegressor", "regression", 42, {"typo": 1})
        with self.assertRaisesRegex(ValueError, "unknown classification model"):
            registry.create_estimator("ModelLRRegressor", "classification", 42)

    def test_capabilities_and_provenance_without_vendor(self):
        with patch.dict(sys.modules, {"ccpl_training_models": None}):
            catalog = ml.capabilities()
            self.assertEqual(catalog["regression"], list(registry.MODELS["regression"]))
            self.assertIn("r2", catalog["metrics"]["regression"])
            self.assertNotIn("PsyTrainer", registry.versions())
            with self.assertRaisesRegex(RuntimeError, "--legacy"):
                ml.capabilities(legacy=True)

    @unittest.skipUnless(importlib.util.find_spec("ccpl_training_models"), "optional original engine not installed")
    def test_original_factory_mapping_and_parameters_match(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from ccpl_training_models.model.model_factory import ModelFactory
            factory = ModelFactory("general")
            for task, models in registry.MODELS.items():
                self.assertEqual(list(models), factory.get_all_model_tags(0 if task == "classification" else 1))
                for tag in models:
                    with self.subTest(tag=tag):
                        original = clone(factory.create_model(tag, {}).model)
                        params = original.get_params(deep=False)
                        updates = {k: 42 for k in ("random_state", "random_seed") if k in params}
                        if "n_jobs" in params:
                            updates["n_jobs"] = 1
                        if tag == "ModelLRClassifier":
                            updates["max_iter"] = 2000
                        original.set_params(**updates)
                        migrated = registry.create_estimator(tag, task, 42)
                        self.assertIs(type(original), type(migrated))
                        self.assertEqual(repr(original.get_params()), repr(migrated.get_params()))


if __name__ == "__main__":
    unittest.main()
