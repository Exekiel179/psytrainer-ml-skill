"""Real configure/train/predict cycles through the compact agent CLI."""

import configparser
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def call(*args, success=True):
    run = subprocess.run([sys.executable, str(ROOT / "scripts/ml.py"), *map(str, args)],
                         text=True, capture_output=True)
    data = json.loads(run.stdout)
    if success:
        assert run.returncode == 0, (data, run.stderr)
    else:
        assert run.returncode != 0, data
    return data, len(run.stdout.encode())


def main():
    with tempfile.TemporaryDirectory(prefix="psytrainer-agent-") as temp:
        root = Path(temp)
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"],
                         index=[f"{i:03}" for i in range(60)])
        x.to_csv(root / "x.csv")
        for task in ("regression", "classification"):
            target = x.a * 2 + x.b if task == "regression" else (x.a > 0).astype(int)
            y = pd.DataFrame({"target": target, "ignored": target})
            y.to_csv(root / "y.csv")
            config = root / f"{task}.ini"
            output = root / task
            configured, _ = call("configure", "--features", root / "x.csv", "--labels", root / "y.csv",
                                 "--task", task, "--target", "target", "--config", config,
                                 "--output-dir", output, "--cv", "3")
            assert configured["profile"]["ready"]
            trained, compact_bytes = call("train", "--config", config, "--top", "1")
            assert len(trained["targets"]) == 1
            assert trained["targets"][0]["models"] == 2
            assert trained["targets"][0]["invalid_scores"] == 0
            assert len(trained["targets"][0]["top"]) == 1
            log_bytes = (output / "run.log").stat().st_size
            assert log_bytes > compact_bytes
            manifest = json.loads((output / "run-manifest.json").read_text())
            assert len(manifest["feature_sha256"]) == 64
            refused, _ = call("train", "--config", config, success=False)
            assert "not empty" in refused["error"]
            settings = configparser.ConfigParser()
            settings.read_dict({"batch_predict": {"feature_file": str(root / "x.csv"),
                "model_root": str(output / "target"), "output_dir": str(root / f"{task}-predictions")}})
            prediction_config = root / f"{task}-predict.ini"
            with prediction_config.open("w") as handle:
                settings.write(handle)
            predicted, _ = call("predict", "--config", prediction_config)
            assert predicted["failures"] == 0 and predicted["models"] == 2
            for model in predicted["preview"]:
                values = pd.read_csv(model["output"], index_col=0)
                assert len(values) == 60 and values["prediction"].notna().all()
            print(json.dumps({"task": task, "samples": 60, "models": 2,
                              "compact_stdout_bytes": compact_bytes, "full_log_bytes": log_bytes,
                              "stdout_reduction_percent": round(100 * (1 - compact_bytes / log_bytes), 1)}))


if __name__ == "__main__":
    main()
