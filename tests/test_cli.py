from __future__ import annotations

import configparser
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"test_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer_cli = load_script("PsyTrainer")
predict_cli = load_script("batch_predict")


class FakeTrainer:
    labels_seen: dict[str, list[int]] = {}

    def __init__(self, train_type: str):
        self.train_type = train_type

    def set_base_config(self, **settings):
        self.settings = settings

    def set_scoring(self, scoring):
        self.scoring = scoring

    def run(self, features, labels, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.__class__.labels_seen[str(labels.name)] = labels.tolist()
        return {"model": "fixture"}, {"score": 0.5}


class FakeModel:
    classes_ = [0, 1]

    def predict(self, frame):
        return [1] * len(frame)

    def predict_proba(self, frame):
        import numpy as np

        return np.array([[0.2, 0.8]] * len(frame))


def write_config(path: Path, sections: dict[str, dict[str, str]]) -> None:
    parser = configparser.ConfigParser()
    parser.read_dict(sections)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)


class PsyTrainerCliTests(unittest.TestCase):
    def test_family_tags_remain_strings_for_vendor_api(self):
        self.assertEqual(trainer_cli.parse_tags("classification", {"classification"}), "classification")
        self.assertEqual(trainer_cli.parse_tags("all_ff", {"all_ff"}), "all_ff")
        self.assertEqual(trainer_cli.parse_tags("all_resample", {"all_resample"}), "all_resample")
        self.assertEqual(trainer_cli.parse_tags("ModelA,ModelB", {"classification"}), ["ModelA", "ModelB"])

    def test_training_aligns_labels_by_sample_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features.csv"
            labels = root / "labels.csv"
            config = root / "ml.ini"
            output = root / "training"
            pd.DataFrame({"x": [10, 20]}, index=["a", "b"]).to_csv(features)
            pd.DataFrame({"target": [2, 1]}, index=["b", "a"]).to_csv(labels)
            write_config(config, {"PsyTrainer": {
                "feature_file": str(features),
                "label_file": str(labels),
                "output_dir": str(output),
                "cv_times": "2",
                "pca_num": "0",
            }})
            package = types.ModuleType("ccpl_training_models")
            trainer_module = types.ModuleType("ccpl_training_models.trainer")
            trainer_module.Trainer = FakeTrainer
            package.trainer = trainer_module
            with patch.dict(sys.modules, {
                "ccpl_training_models": package,
                "ccpl_training_models.trainer": trainer_module,
            }):
                code = trainer_cli.main(["--config", str(config)])
            self.assertEqual(code, 0)
            self.assertEqual(FakeTrainer.labels_seen["target"], [1, 2])
            self.assertTrue((output / "training-summary.json").is_file())

    def test_training_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features.csv"
            labels = root / "labels.csv"
            config = root / "ml.ini"
            output = root / "training"
            pd.DataFrame({"x": [1, 2]}, index=["a", "b"]).to_csv(features)
            pd.DataFrame({"target": [0, 1]}, index=["a", "b"]).to_csv(labels)
            write_config(config, {"PsyTrainer": {
                "feature_file": str(features),
                "label_file": str(labels),
                "output_dir": str(output),
            }})
            self.assertEqual(trainer_cli.main(["--config", str(config), "--dry-run"]), 0)
            self.assertFalse(output.exists())

    def test_prediction_rejects_missing_values_without_dropping_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.csv"
            pd.DataFrame({"x": [1.0, None]}, index=["a", "b"]).to_csv(path)
            with self.assertRaisesRegex(ValueError, "no rows were dropped"):
                predict_cli.load_features(path)

    def test_prediction_writes_predictions_and_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features.csv"
            model_dir = root / "models" / "ExampleModel"
            output = root / "predictions"
            config = root / "ml.ini"
            model_dir.mkdir(parents=True)
            pd.DataFrame({"x": [3.0, 4.0]}, index=["a", "b"]).to_csv(features)
            (model_dir / "model.pkl").write_bytes(b"fixture")
            pd.DataFrame({"feature_name": ["0"]}).to_csv(model_dir / "feature_filter.csv")
            write_config(config, {"batch_predict": {
                "feature_file": str(features),
                "model_root": str(model_dir.parent),
                "output_dir": str(output),
            }})
            with patch.object(joblib, "load", return_value=FakeModel()):
                code = predict_cli.main(["--config", str(config)])
            self.assertEqual(code, 0)
            predictions = pd.read_csv(output / "ExampleModel.csv", index_col=0)
            self.assertEqual(predictions["prediction"].tolist(), [1, 1])
            self.assertIn("probability_1", predictions.columns)
            self.assertTrue((output / "prediction-summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
