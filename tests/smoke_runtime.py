"""Explicit end-to-end test using the installed, real PsyTrainer distribution."""

import configparser
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="psytrainer-smoke-") as temp:
        root = Path(temp)
        rng = np.random.default_rng(42)
        values = rng.normal(size=(30, 3))
        pd.DataFrame(values, columns=["a", "b", "c"]).to_csv(root / "features.csv")
        pd.DataFrame({"target": values[:, 0] * 2 + values[:, 1]}).to_csv(root / "labels.csv")
        config = configparser.ConfigParser()
        config.read_dict({"PsyTrainer": {
            "feature_file": "features.csv", "label_file": "labels.csv",
            "output_dir": "training", "cv_times": "3", "pca_num": "2",
            "ff_tags": "FFByFregression", "model_tags": "ModelLRRegressor",
            "scoring": "mae", "is_use_model_params": "false",
        }, "batch_predict": {
            "feature_file": "features.csv", "model_root": "training/target", "output_dir": "predictions",
        }})
        path = root / "ml.ini"
        with path.open("w") as handle:
            config.write(handle)
        for name in ("PsyTrainer.py", "batch_predict.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--config", str(path)], check=True)
        training = json.loads((root / "training/training-summary.json").read_text())
        prediction = json.loads((root / "predictions/prediction-summary.json").read_text())
        assert training["sample_count"] == 30
        assert training["targets"][0]["results"]
        assert prediction["failure_count"] == 0
        assert prediction["models"][0]["rows"] == 30
        predicted = pd.read_csv(prediction["models"][0]["output"], index_col=0)
        assert np.isfinite(predicted["prediction"]).all()
        print("Real PsyTrainer training and prediction passed (30 samples).")


if __name__ == "__main__":
    main()
