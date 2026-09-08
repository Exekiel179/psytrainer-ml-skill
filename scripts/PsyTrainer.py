#!/usr/bin/env python3
"""Train one or more targets with the external PsyTrainer framework."""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path
from typing import Any


SECTION = "PsyTrainer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tabular models with PsyTrainer")
    parser.add_argument("--config", type=Path, default=Path("config/ml.ini"))
    parser.add_argument("--output-dir", type=Path, help="Override output_dir in ml.ini")
    parser.add_argument("--target", action="append", default=[], help="Train only this label column; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without training")
    return parser


def parse_list(value: str | None) -> list[str] | None:
    items = [item.strip() for item in (value or "").split(",") if item.strip()]
    return items or None


def resolve_path(config_path: Path, value: str, name: str) -> Path:
    if not value.strip():
        raise ValueError(f"{name} is required in [{SECTION}]")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_settings(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    parser = configparser.ConfigParser()
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    if not parser.has_section(SECTION):
        raise ValueError(f"configuration is missing [{SECTION}]")

    section = parser[SECTION]
    feature_file = resolve_path(config_path, section.get("feature_file", ""), "feature_file")
    label_file = resolve_path(config_path, section.get("label_file", ""), "label_file")
    output_dir = output_override.expanduser().resolve() if output_override else resolve_path(
        config_path, section.get("output_dir", ""), "output_dir"
    )
    if not feature_file.is_file():
        raise FileNotFoundError(f"feature_file not found: {feature_file}")
    if not label_file.is_file():
        raise FileNotFoundError(f"label_file not found: {label_file}")

    settings: dict[str, Any] = {
        "feature_file": feature_file,
        "label_file": label_file,
        "output_dir": output_dir,
        "train_type": section.get("train_type", "general").strip() or "general",
        "cv_times": section.getint("cv_times", fallback=5),
        "pca_num": section.getint("pca_num", fallback=0),
        "is_need_clean": section.getboolean("is_need_clean", fallback=True),
        "is_use_model_params": section.getboolean("is_use_model_params", fallback=False),
        "scoring": parse_list(section.get("scoring")),
        "ff_tags": parse_list(section.get("ff_tags")),
        "model_tags": parse_list(section.get("model_tags")),
        "resample_tags": parse_list(section.get("resample_tags")),
    }
    if settings["cv_times"] < 2:
        raise ValueError("cv_times must be at least 2")
    if settings["pca_num"] < 0:
        raise ValueError("pca_num cannot be negative")
    return settings


def load_tables(feature_file: Path, label_file: Path, selected_targets: list[str]):
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("PsyTrainer requires pandas and numpy") from exc

    features = pd.read_csv(feature_file, index_col=0)
    labels = pd.read_csv(label_file, index_col=0)
    if features.empty or labels.empty:
        raise ValueError("feature and label CSV files must not be empty")
    if features.index.has_duplicates or labels.index.has_duplicates:
        raise ValueError("sample IDs must be unique")
    if features.columns.has_duplicates or labels.columns.has_duplicates:
        raise ValueError("column names must be unique")

    features.index = features.index.map(str)
    labels.index = labels.index.map(str)
    if set(features.index) != set(labels.index):
        missing_labels = sorted(set(features.index) - set(labels.index))[:5]
        missing_features = sorted(set(labels.index) - set(features.index))[:5]
        raise ValueError(
            "feature and label sample IDs differ; "
            f"missing labels={missing_labels}, missing features={missing_features}"
        )
    labels = labels.loc[features.index]

    try:
        features = features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("all feature columns must be numeric") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("features contain missing or non-finite values")

    available = [str(column) for column in labels.columns if str(column) != "index"]
    targets = selected_targets or available
    unknown = sorted(set(targets) - set(available))
    if unknown:
        raise ValueError(f"unknown target columns: {unknown}")
    if not targets:
        raise ValueError("label_file contains no target columns")
    for target in targets:
        if labels[target].isna().any():
            raise ValueError(f"target '{target}' contains missing values")
    return features, labels, targets


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return str(value)


def safe_target_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"target cannot be used as a directory name: {name!r}")
    return name


def train(settings: dict[str, Any], selected_targets: list[str]) -> dict[str, Any]:
    try:
        from ccpl_training_models.trainer import Trainer
    except ImportError as exc:
        raise RuntimeError(
            "PsyTrainer is unavailable; use Python 3.13 with the PsyTrainer wheel installed"
        ) from exc

    features, labels, targets = load_tables(
        settings["feature_file"], settings["label_file"], selected_targets
    )
    output_dir: Path = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for target in targets:
        target_dir = output_dir / safe_target_name(target)
        trainer = Trainer(settings["train_type"])
        trainer.set_base_config(
            cv_times=settings["cv_times"],
            pca_num=settings["pca_num"],
            is_need_clean=settings["is_need_clean"],
            is_use_model_params=settings["is_use_model_params"],
            ff_tags=settings["ff_tags"],
            model_tags=settings["model_tags"],
            resample_tags=settings["resample_tags"],
        )
        if settings["scoring"]:
            trainer.set_scoring(settings["scoring"])
        best, results = trainer.run(features, labels.loc[:, target], str(target_dir))
        summaries.append({
            "target": target,
            "output_directory": str(target_dir),
            "best": json_value(best),
            "results": json_value(results),
        })

    summary = {
        "sample_count": len(features),
        "feature_count": len(features.columns),
        "targets": summaries,
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def dry_run(settings: dict[str, Any], selected_targets: list[str]) -> dict[str, Any]:
    features, _, targets = load_tables(
        settings["feature_file"], settings["label_file"], selected_targets
    )
    return {
        "operation": "train",
        "feature_file": str(settings["feature_file"]),
        "label_file": str(settings["label_file"]),
        "output_dir": str(settings["output_dir"]),
        "sample_count": len(features),
        "feature_count": len(features.columns),
        "targets": targets,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config, args.output_dir)
        result = dry_run(settings, args.target) if args.dry_run else train(settings, args.target)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
