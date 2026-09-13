import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ml
from table_io import read_table


class MlTests(unittest.TestCase):
    def tables(self, root):
        ids = [f"{i:03}" for i in range(12)]
        x, y = root / "x.csv", root / "y.csv"
        pd.DataFrame({"x": range(12), "constant": [1] * 12}, index=ids).to_csv(x)
        pd.DataFrame({"a": [0, 1] * 6, "b": range(12)}, index=ids).to_csv(y)
        return x, y

    def test_ids_preserved_and_duplicate_headers_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x, _ = self.tables(root)
            self.assertEqual(read_table(x).index[0], "000")
            x.write_text("id,x,x\n001,1,2\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_table(x)

    def test_inspect_reports_invalid_cells_and_class_issues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x, y = self.tables(root)
            profile = ml.inspect_data(x, y, ["a"], "classification", 3)
            self.assertTrue(profile["ready"])
            self.assertEqual(profile["constant_feature_count"], 1)
            self.assertNotIn("000", json.dumps(profile))
            frame = read_table(x)
            frame.loc["000", "x"] = float("nan")
            frame.to_csv(x)
            profile = ml.inspect_data(x, y, ["b"], "classification", 3)
            self.assertFalse(profile["ready"])
            self.assertEqual(profile["invalid_feature_cells"], 1)
            self.assertTrue(any("0 and 1" in error for error in profile["errors"]))

    def test_configuration_preserves_target_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x, y = self.tables(root)
            args = ml.parser().parse_args(["configure", "--features", str(x), "--labels", str(y),
                                          "--task", "regression", "--target", "b", "--cv", "3",
                                          "--config", str(root / "run.ini"),
                                          "--output-dir", str(root / "out")])
            catalog = {"regression": ml.PRESETS["regression"]["quick"], "metrics": {"regression": ["mae"]}}
            with patch.object(ml, "capabilities", return_value=catalog):
                value, code = ml.configure(args)
                self.assertEqual(code, 0)
                self.assertEqual(value["estimated_fits"], 8)
                settings = ml.training.load_settings(args.config)
                self.assertEqual(settings["targets"], ["b"])
                self.assertIsNone(settings["ff_tags"])
                self.assertFalse(settings["is_need_clean"])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    ml.configure(args)

    def test_report_bounds_rankings_and_excludes_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text(json.dumps({"sample_count": 10, "feature_count": 2,
                "targets": [{"target": "a", "results": [
                    {"tag": "bad", "best_score": float("nan")},
                    {"tag": "second", "best_score": -2},
                    {"tag": "best", "best_score": -1}]}]}))
            result = ml.report(path, 1)
            self.assertEqual(result["targets"][0]["invalid_scores"], 1)
            self.assertEqual(result["targets"][0]["top"], [{"model": "best", "cv_score": -1}])

    def test_unfiltered_model_uses_saved_order_and_rejects_missing(self):
        model = type("Model", (), {"feature_names_in_": ["b", "a"]})()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "NoneType&ModelLRRegressor"
            directory.mkdir()
            frame = pd.DataFrame({"a": [1], "b": [2]})
            result = ml.prediction.selected_input(frame, directory, None, model)
            self.assertEqual(list(result.columns), ["b", "a"])
            with self.assertRaisesRegex(ValueError, "missing model features"):
                ml.prediction.selected_input(frame[["a"]], directory, None, model)
            with self.assertRaisesRegex(FileNotFoundError, "feature_filter"):
                ml.prediction.selected_input(frame, Path(temp), None, model)

    def test_unknown_report_returns_structured_failure(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ml.main(["report", "/nonexistent/summary.json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(buffer.getvalue())["status"], "failed")

    def test_subprocess_failure_is_compact_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x, _ = self.tables(root)
            model_root = root / "models"
            model = model_root / "broken"
            model.mkdir(parents=True)
            (model / "model.pkl").write_bytes(b"not a valid model")
            config = root / "predict.ini"
            config.write_text(f"[batch_predict]\nfeature_file = {x}\nmodel_root = {model_root}\noutput_dir = {root / 'out'}\n")
            args = ml.parser().parse_args(["predict", "--config", str(config)])
            with patch.object(ml.importlib.metadata, "version", return_value="test"):
                result, code = ml.execute(args)
            self.assertNotEqual(code, 0)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(Path(result["log"]).is_file())
            self.assertTrue(Path(result["summary"]).is_file())
            self.assertLess(len(json.dumps(result)), 1000)
            manifest = json.loads((root / "out/run-manifest.json").read_text())
            self.assertEqual(manifest["exit_code"], code)


if __name__ == "__main__":
    unittest.main()
