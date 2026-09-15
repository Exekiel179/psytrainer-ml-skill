#!/usr/bin/env python3
"""Compact, structured operations for the PsyTrainer Skill."""

from __future__ import annotations

import argparse
import configparser
import contextlib
import hashlib
import io
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

import PsyTrainer as training
import batch_predict as prediction
from table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "classification": {
        "quick": ["ModelLRClassifier", "ModelDecisionTreesClassifier"],
        "standard": ["ModelLRClassifier", "ModelRandomForestClassifier", "ModelLGBMClassifier"],
    },
    "regression": {
        "quick": ["ModelLRRegressor", "ModelExtraTreeRegressor"],
        "standard": ["ModelLRRegressor", "ModelRandomForestRegressor", "ModelLGBMRegressor"],
    },
}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def capabilities(legacy=False):
    if not legacy:
        from model_registry import MODELS
        from pipeline_train import SCORERS, DEFAULT_MODELS
        from pipeline_options import SAMPLERS, DOMAIN_MODELS
        return {"backend": "local-registry/v1", **{task: list(models) for task, models in MODELS.items()},
                "metrics": {task: list(metrics) for task, metrics in SCORERS.items()},
                "defaults": DEFAULT_MODELS, "model_parameters": "pipeline_train.py train --model-params FILE.json",
                "search": ["grid", "random"], "search_space": "--search-space FILE.json",
                "feature_filters": ["minmax", "kbest", "f-threshold", "forward", "pca"],
                "resamplers": list(SAMPLERS), "domain_presets": DOMAIN_MODELS,
                "resume": "--resume reuses CV folds with identical data/config/code/versions",
                "legacy": "Historical INI only: explicitly supply --legacy --wheel /path/to/PsyTrainer.whl; use runtime-legacy.json python"}
    # Keep original INI capabilities separate from Pipeline scoring and preprocessing.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from ccpl_training_models.model.model_factory import ModelFactory
        except ImportError as exc:
            raise RuntimeError("Historical INI engine requires an external package: scripts/install_runtime.py --legacy --wheel /path/to/PsyTrainer.whl; new analyses use pipeline_train.py") from exc
        from ccpl_training_models.feature.ff_factory import FFFactory
        from ccpl_training_models.sampler.sampler_factory import SamplerFactory
        from ccpl_training_models.util.scoring_utils import SCORINGS_CLASSIFIER, SCORINGS_DEFAULT
        factory = ModelFactory("general")
    return {
        "classification": factory.get_all_model_tags(0),
        "regression": factory.get_all_model_tags(1),
        "feature_filters": FFFactory.get_all_ff_tags(),
        "resamplers": SamplerFactory.get_all_sampler_tags(),
        "metrics": {"classification": [s.name for s in SCORINGS_CLASSIFIER],
                    "regression": [s.name for s in SCORINGS_DEFAULT]},
        "presets": PRESETS,
        "pipeline_runner": {"script": "scripts/pipeline_train.py",
                            "reference": "references/pipeline.md",
                            "supports": ["fold-local preprocessing", "group/time CV", "held-out evaluation",
                                         "permutation importance", "figures", "Word reports"],
                            "note": "The following limits describe the legacy INI runner only."},
        "limits": ["numeric tabular features only", "no grouped or temporal CV in this wrapper",
                   "feature filtering and resampling happen before CV",
                   "vendor random folds are not reproducibly seeded",
                   "vendor r2 scorer negates R2; do not use it for model selection"],
    }


def inspect_data(features_path, labels_path, targets, task, cv):
    import numpy as np
    import pandas as pd
    features = read_table(features_path)
    labels = read_table(labels_path)
    if features.empty or labels.empty:
        raise ValueError("feature and label CSVs must not be empty")
    if set(features.index) != set(labels.index):
        raise ValueError("feature and label sample IDs differ")
    labels = labels.loc[features.index]
    targets = targets or list(labels.columns)
    if not targets or set(targets) - set(labels.columns):
        raise ValueError("unknown or empty target selection")
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate target selection")
    for target in targets:
        training.safe_target_name(target)
        if "," in target:
            raise ValueError("target names cannot contain commas in generated INI settings")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    problems = []
    if invalid.any():
        problems.append("features contain missing, nonnumeric or non-finite values")
    if not 2 <= cv <= len(features):
        problems.append("cv must be between 2 and sample count")
    constant = list(features.columns[features.nunique(dropna=False) <= 1])
    warnings = []
    if constant:
        warnings.append("constant features present")
    if len(features.columns) >= len(features):
        warnings.append("feature count is at least sample count; high overfitting risk")
    target_info = []
    for name in targets:
        column = labels[name]
        unique = int(column.nunique())
        item = {"target": name, "unique": unique, "missing": int(column.isna().sum())}
        if column.isna().any() or unique < 2:
            problems.append(f"target {name}: missing values or fewer than two distinct values")
        if task == "regression":
            converted = pd.to_numeric(column, errors="coerce")
            if not np.isfinite(converted.to_numpy(dtype=float)).all():
                problems.append(f"target {name}: regression requires finite numeric values")
        else:
            # Vendor scorers use binary precision/recall/F1 with pos_label=1.
            if unique != 2 or set(column.dropna().unique()) != {0, 1}:
                problems.append(f"target {name}: classification presets require labels encoded as 0 and 1")
            counts = column.value_counts()
            item["class_counts"] = {str(k): int(v) for k, v in counts.head(10).items()}
            if not counts.empty and int(counts.min()) < cv:
                problems.append(f"target {name}: minority class has fewer samples than CV folds")
            if not counts.empty and int(counts.max()) > 4 * int(counts.min()):
                warnings.append(f"target {name}: class imbalance; inspect class-specific performance")
        target_info.append(item)
    if task == "classification":
        warnings.append("vendor uses random KFold, not stratified folds; folds can still miss a class")
    return {"ready": not problems, "samples": len(features), "features": len(features.columns),
            "invalid_feature_cells": int(invalid.sum()), "constant_features": constant[:20],
            "constant_feature_count": len(constant), "targets": target_info,
            "errors": problems, "warnings": warnings}


def configure(args):
    profile = inspect_data(args.features, args.labels, args.target, args.task, args.cv)
    if not profile["ready"]:
        return profile, 2
    catalog = capabilities(legacy=True)
    models = args.model or PRESETS[args.task][args.preset]
    if set(models) - set(catalog[args.task]):
        raise ValueError(f"unknown {args.task} models; use capabilities")
    metric = args.metric or ("accuracy" if args.task == "classification" else "mae")
    if metric not in catalog["metrics"][args.task] or metric == "r2":
        raise ValueError("unsupported selection metric; use capabilities (r2 selection is disabled)")
    if args.config.exists():
        raise ValueError("config already exists; choose a new path to preserve previous settings")
    output = args.output_dir.expanduser().resolve()
    config = configparser.ConfigParser()
    config.read_dict({"PsyTrainer": {
        "feature_file": str(args.features.expanduser().resolve()),
        "label_file": str(args.labels.expanduser().resolve()),
        "output_dir": str(output), "train_type": "general", "cv_times": str(args.cv),
        "pca_num": "0", "is_need_clean": "false", "is_use_model_params": "false",
        "ff_tags": "", "model_tags": ",".join(models), "resample_tags": "", "scoring": metric,
        "targets": ",".join(item["target"] for item in profile["targets"]),
    }})
    args.config.parent.mkdir(parents=True, exist_ok=True)
    with args.config.open("x", encoding="utf-8") as handle:
        config.write(handle)
    return {"config": str(args.config.resolve()), "profile": profile, "models": models,
            "metric": metric, "estimated_fits": len(models) * len(profile["targets"]) * (args.cv + 1),
            "note": "CV fits plus final fit estimate; no grid search or preprocessing"}, 0


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def report(path, top):
    if top < 1 or top > 20:
        raise ValueError("top must be between 1 and 20")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "targets" not in data:
        return {"summary": str(Path(path).resolve()), "samples": data["sample_count"],
                "models": data["model_count"], "failures": data["failure_count"],
                "preview": data["models"][:top]}
    targets = []
    for target in data["targets"]:
        results = target["results"]
        if not isinstance(results, list):
            raise ValueError("expected vendor result list")
        valid = [r for r in results if isinstance(r.get("best_score"), (float, int))
                 and math.isfinite(r["best_score"])]
        valid.sort(key=lambda r: r["best_score"], reverse=True)
        targets.append({"target": target["target"], "models": len(results),
                        "invalid_scores": len(results) - len(valid),
                        "top": [{"model": r.get("tag", r.get("ff_and_model")),
                                 "cv_score": r["best_score"]} for r in valid[:top]]})
    manifest_path = Path(path).parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {"summary": str(Path(path).resolve()), "samples": data["sample_count"],
            "features": data["feature_count"], "targets": targets,
            "primary_metric": manifest.get("primary_metric", "unknown; check original config"),
            "interpretation": "within-target vendor CV scores; higher is better, losses are negated; not held-out accuracy"}


def execute(args):
    warnings = []
    if args.operation == "train":
        settings = training.load_settings(args.config, args.output_dir)
        targets = args.target or settings.get("targets", [])
        training.dry_run(settings, targets)
        catalog = capabilities(legacy=True)
        models = settings["model_tags"]
        if isinstance(models, str):
            models = catalog.get(models, [])
        if not models or set(models) - set(catalog["classification"] + catalog["regression"]):
            raise ValueError("unknown model tags; use capabilities")
        families = {"classification" if m in catalog["classification"] else "regression" for m in models}
        if len(families) != 1:
            raise ValueError("run classification and regression in separate configurations")
        task = families.pop()
        profile = inspect_data(settings["feature_file"], settings["label_file"], targets, task, settings["cv_times"])
        if not profile["ready"]:
            return profile, 2
        warnings.extend(profile["warnings"])
        for field, key, all_tag in (("ff_tags", "feature_filters", "all_ff"),
                                    ("resample_tags", "resamplers", "all_resample")):
            tags = settings[field]
            if tags and tags != all_tag and (not isinstance(tags, list) or set(tags) - set(catalog[key])):
                raise ValueError(f"unknown {field}; use capabilities")
        if settings["scoring"] and set(settings["scoring"]) - set(catalog["metrics"][task]):
            raise ValueError("invalid metrics for the selected model family")
        if settings["scoring"] and settings["scoring"][0] == "r2":
            raise ValueError("vendor negates R2; select models using another primary metric")
        if settings["ff_tags"] or settings["resample_tags"]:
            warnings.append("preprocessing/resampling occurs before CV; scores may be optimistic")
        if settings["ff_tags"] == "all_ff" or "FFByPca" in (settings["ff_tags"] or []):
            if not 1 <= settings["pca_num"] <= min(profile["samples"], profile["features"]):
                raise ValueError("PCA requires pca_num between 1 and min(samples, features)")
        command = [sys.executable, str(ROOT / "scripts/PsyTrainer.py"), "--config", str(args.config.resolve())]
        for target in targets:
            command.extend(["--target", target])
        summary_name = "training-summary.json"
    else:
        settings = prediction.load_settings(args.config, args.output_dir)
        check = prediction.dry_run(settings, args.model)
        if any(not c["has_model"] for c in check["models"]):
            raise ValueError("selected model directory is missing model.pkl")
        command = [sys.executable, str(ROOT / "scripts/batch_predict.py"), "--config", str(args.config.resolve())]
        for model in args.model:
            command.extend(["--model", model])
        summary_name = "prediction-summary.json"
    output = settings["output_dir"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose --output-dir for a new run")
    output.mkdir(parents=True, exist_ok=True)
    command.extend(["--output-dir", str(output)])
    log = output / "run.log"
    installed_versions = {}
    for name in ("PsyTrainer", "numpy", "pandas", "scikit-learn"):
        try:
            installed_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    provenance = {"operation": args.operation, "python": sys.executable,
                  "config_sha256": digest(args.config), "feature_sha256": digest(settings["feature_file"]),
                  "command": command, "warnings": warnings,
                  "versions": installed_versions}
    if args.operation == "train":
        provenance["label_sha256"] = digest(settings["label_file"])
        provenance["primary_metric"] = (settings["scoring"] or
                                        (["f1"] if task == "classification" else ["pearson_r"]))[0]
        provenance["cv_folds"] = settings["cv_times"]
    # A subprocess captures native-library output too, without buffering full logs in RAM.
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    provenance["exit_code"] = completed.returncode
    (output / "run-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    summary = output / summary_name
    if completed.returncode:
        return {"status": "failed", "exit_code": completed.returncode, "log": str(log),
                "summary": str(summary) if summary.exists() else None}, completed.returncode
    return {"status": "completed", "log": str(log), "warnings": warnings,
            **report(summary, args.top)}, 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="operation", required=True)
    cap = commands.add_parser("capabilities", help="List Pipeline algorithms and metrics")
    cap.add_argument("--legacy", action="store_true", help="Query original INI engine instead")
    for name in ("inspect", "configure"):
        sub = commands.add_parser(name)
        sub.add_argument("--features", type=Path, required=True)
        sub.add_argument("--labels", type=Path, required=True)
        sub.add_argument("--target", action="append", default=[])
        sub.add_argument("--task", choices=PRESETS, required=True)
        sub.add_argument("--cv", type=int, default=5)
        if name == "configure":
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--preset", choices=("quick", "standard"), default="quick")
            sub.add_argument("--model", action="append", default=[])
            sub.add_argument("--metric")
    for name in ("train", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--target" if name == "train" else "--model", action="append", default=[])
        sub.add_argument("--top", type=int, choices=range(1, 21), default=3)
    sub = commands.add_parser("report")
    sub.add_argument("summary", type=Path)
    sub.add_argument("--top", type=int, choices=range(1, 21), default=3)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if __name__ == "__main__" and args.operation in {"inspect", "configure", "capabilities"} and not getattr(args, "legacy", False):
        from runtime_dependencies import CORE, bootstrap
        bootstrap(__file__, CORE if args.operation == "capabilities" else ("numpy", "pandas"))
    try:
        if args.operation == "capabilities":
            value, code = capabilities(legacy=args.legacy), 0
        elif args.operation == "inspect":
            value = inspect_data(args.features, args.labels, args.target, args.task, args.cv)
            code = 0 if value["ready"] else 2
        elif args.operation == "configure":
            value, code = configure(args)
        elif args.operation == "report":
            value, code = report(args.summary, args.top), 0
        else:
            value, code = execute(args)
        emit(value)
        return code
    except Exception as exc:
        emit({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
