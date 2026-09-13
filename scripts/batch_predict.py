#!/usr/bin/env python3
"""Apply PsyTrainer model directories to a feature CSV."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable
from table_io import read_table


SECTION = "batch_predict"
NAMED_FEATURE_PREFIXES = ("FFByFregression", "FFByStepForward")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-predict with PsyTrainer models")
    parser.add_argument("--config", type=Path, default=Path("config/ml.ini"))
    parser.add_argument("--output-dir", type=Path, help="Override output_dir in ml.ini")
    parser.add_argument("--model", action="append", default=[], help="Use only this model directory; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading models")
    return parser


def resolve_path(config_path: Path, value: str, name: str) -> Path:
    if not value.strip():
        raise ValueError(f"{name} is required in [{SECTION}]")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_settings(config_path: Path, output_override: Path | None = None) -> dict[str, Path]:
    config_path = config_path.expanduser().resolve()
    parser = configparser.ConfigParser()
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    if not parser.has_section(SECTION):
        raise ValueError(f"configuration is missing [{SECTION}]")

    section = parser[SECTION]
    feature_file = resolve_path(config_path, section.get("feature_file", ""), "feature_file")
    model_root = resolve_path(config_path, section.get("model_root", ""), "model_root")
    output_dir = output_override.expanduser().resolve() if output_override else resolve_path(
        config_path, section.get("output_dir", ""), "output_dir"
    )
    if not feature_file.is_file():
        raise FileNotFoundError(f"feature_file not found: {feature_file}")
    if not model_root.is_dir():
        raise FileNotFoundError(f"model_root not found: {model_root}")
    return {"feature_file": feature_file, "model_root": model_root, "output_dir": output_dir}


def model_directories(model_root: Path, selected: list[str]) -> list[Path]:
    directories = sorted(
        (path for path in model_root.iterdir() if path.is_dir()), key=lambda path: path.name
    )
    if selected:
        wanted = set(selected)
        unknown = sorted(wanted - {path.name for path in directories})
        if unknown:
            raise ValueError(f"unknown model directories: {unknown}")
        directories = [path for path in directories if path.name in wanted]
    if not directories:
        raise ValueError("model_root contains no selected model directories")
    return directories


def load_features(path: Path):
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("prediction requires pandas and numpy") from exc

    data = read_table(path)
    if data.empty:
        raise ValueError("feature CSV must not be empty")
    if data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError("sample IDs and feature names must be unique")
    try:
        data = data.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("all feature columns must be numeric") from exc
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("features contain missing or non-finite values; no rows were dropped")
    return data


def selected_input(data, model_dir: Path, loader: Callable[[Path], Any], model=None):
    import pandas as pd

    feature_path = model_dir / "feature_filter.csv"
    if not feature_path.is_file():
        if model_dir.name.startswith("NoneType&") and model is not None:
            names = getattr(model, "feature_names_in_", None)
            if names is not None:
                names = [str(name) for name in names]
                if not names or len(names) != len(set(names)):
                    raise ValueError("invalid model feature names")
                missing = [name for name in names if name not in data.columns]
                if missing:
                    raise ValueError(f"input is missing model features: {missing[:10]}")
                return data.loc[:, names]
        raise FileNotFoundError("missing feature_filter.csv")
    feature_frame = pd.read_csv(feature_path, index_col=0)
    if "feature_name" not in feature_frame.columns:
        raise ValueError("feature_filter.csv lacks feature_name column")
    selected = [str(value) for value in feature_frame["feature_name"].tolist()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("feature_filter.csv contains empty or duplicate feature names")

    base = data.copy()
    if model_dir.name.startswith(NAMED_FEATURE_PREFIXES):
        base.columns = [str(column) for column in base.columns]
    else:
        base.columns = [str(index) for index in range(base.shape[1])]

    pca_path = model_dir / "pca" / "pca.m"
    if pca_path.is_file():
        transformed = loader(pca_path).transform(base)
        base = pd.DataFrame(
            transformed,
            index=data.index,
            columns=[str(index) for index in range(transformed.shape[1])],
        )

    missing = [name for name in selected if name not in base.columns]
    if missing:
        raise ValueError(f"input is missing selected features: {missing[:10]}")
    return base.loc[:, selected]


def safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return value or "model"


def predict(settings: dict[str, Path], selected_models: list[str]) -> dict[str, Any]:
    try:
        import pandas as pd
        from joblib import load
    except ImportError as exc:
        raise RuntimeError("prediction requires pandas and joblib") from exc

    data = load_features(settings["feature_file"])
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for model_dir in model_directories(settings["model_root"], selected_models):
        record: dict[str, Any] = {"model": model_dir.name}
        try:
            model_path = model_dir / "model.pkl"
            if not model_path.is_file():
                raise FileNotFoundError("missing model.pkl")
            model = load(model_path)
            model_input = selected_input(data, model_dir, load, model)
            values = model.predict(model_input)
            values = values.tolist() if hasattr(values, "tolist") else list(values)
            if len(values) != len(data):
                raise ValueError(f"model returned {len(values)} predictions for {len(data)} samples")

            output = pd.DataFrame(index=data.index)
            if values and isinstance(values[0], (list, tuple)):
                for index in range(len(values[0])):
                    output[f"prediction_{index}"] = [row[index] for row in values]
            else:
                output["prediction"] = values
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(model_input)
                if getattr(probabilities, "ndim", 0) == 2:
                    classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
                    for index, class_name in enumerate(classes):
                        output[f"probability_{class_name}"] = probabilities[:, index]

            stem = safe_filename(model_dir.name)
            if stem in used_names:
                raise ValueError("model names collide after filename normalization")
            used_names.add(stem)
            output_path = output_dir / f"{stem}.csv"
            output.to_csv(output_path)
            record.update({"status": "completed", "rows": len(output), "output": str(output_path)})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
        rows.append(record)

    summary = {
        "sample_count": len(data),
        "model_count": len(rows),
        "failure_count": sum(row["status"] == "failed" for row in rows),
        "models": rows,
    }
    (output_dir / "prediction-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def dry_run(settings: dict[str, Path], selected_models: list[str]) -> dict[str, Any]:
    data = load_features(settings["feature_file"])
    directories = model_directories(settings["model_root"], selected_models)
    checks = []
    for directory in directories:
        checks.append({
            "model": directory.name,
            "has_model": (directory / "model.pkl").is_file(),
            "has_feature_filter": (directory / "feature_filter.csv").is_file(),
            "has_pca": (directory / "pca" / "pca.m").is_file(),
        })
    return {
        "operation": "predict",
        "feature_file": str(settings["feature_file"]),
        "model_root": str(settings["model_root"]),
        "output_dir": str(settings["output_dir"]),
        "sample_count": len(data),
        "models": checks,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config, args.output_dir)
        result = dry_run(settings, args.model) if args.dry_run else predict(settings, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if not args.dry_run and result["failure_count"] else 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
