#!/usr/bin/env python3
"""Leakage-aware model comparison, held-out evaluation and portable prediction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score, get_scorer)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedKFold, TimeSeriesSplit, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from table_io import read_table
from model_registry import MODELS, create_estimator, versions

SCORERS = {
    "classification": {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
                       "f1": "f1_macro", "roc_auc": "roc_auc"},
    "regression": {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                   "r2": "r2"},
}
DEFAULT_MODELS = {
    "classification": ["ModelLRClassifier", "ModelRandomForestClassifier"],
    "regression": ["ModelLRRegressor", "ModelRandomForestRegressor"],
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_xy(features, labels, target, task, impute):
    x, labels = read_table(features), read_table(labels)
    if x.empty or set(x.index) != set(labels.index):
        raise ValueError("features and labels require the same nonempty sample IDs")
    if target not in labels:
        raise ValueError(f"unknown target: {target}")
    y = labels.loc[x.index, target]
    x = x.apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(x.to_numpy()).any() or (impute == "none" and x.isna().any().any()):
        raise ValueError("non-finite features; explicitly choose --impute median for missing values")
    if y.isna().any():
        raise ValueError("target has missing values")
    if task == "regression":
        y = pd.to_numeric(y, errors="raise").astype(float)
        if not np.isfinite(y).all():
            raise ValueError("regression labels must be finite")
    else:
        y = y.astype(str)
    return x, y


def load_metadata(path, index, args):
    if args.split == "random":
        if path or args.group_column or args.time_column:
            raise ValueError("metadata requires --split group or time")
        return None
    column = args.group_column if args.split == "group" else args.time_column
    if not path or not column:
        raise ValueError(f"{args.split} splitting requires metadata and its column name")
    frame = read_table(path)
    if set(frame.index) != set(index) or column not in frame:
        raise ValueError("metadata must match sample IDs and contain the split column")
    values = frame.loc[index, column]
    if values.isna().any():
        raise ValueError("split metadata has missing values")
    if args.split == "time":
        # Numeric times remain numeric; ISO date/time strings become UTC timestamps.
        if not pd.api.types.is_numeric_dtype(values):
            values = pd.to_datetime(values, utc=True, errors="raise")
        elif not np.isfinite(values).all():
            raise ValueError("time values must be finite")
    else:
        values = values.astype(str)
    return values


def holdout_indices(x, y, meta, args):
    ids = np.arange(len(x))
    if args.split == "random":
        train, test = train_test_split(ids, test_size=args.test_size, random_state=args.seed,
                                      stratify=y if args.task == "classification" else None)
    elif args.split == "group":
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                            random_state=args.seed).split(x, y, meta))
    else:
        units = np.sort(meta.unique())
        boundary = len(units) - max(1, int(np.ceil(len(units) * args.test_size)))
        if boundary - args.gap < 1:
            raise ValueError("too few distinct times for holdout and gap")
        train = ids[meta.isin(units[:boundary - args.gap])]
        test = ids[meta.isin(units[boundary:])]
    return np.asarray(train), np.asarray(test)


def cv_indices(x, y, meta, args):
    if args.split == "random":
        splitter = (StratifiedKFold(args.cv, shuffle=True, random_state=args.seed)
                    if args.task == "classification" else
                    KFold(args.cv, shuffle=True, random_state=args.seed))
        folds = list(splitter.split(x, y))
    elif args.split == "group":
        folds = list(GroupKFold(args.cv).split(x, y, meta))
    else:
        units = np.sort(meta.unique())
        folds = [(np.flatnonzero(meta.isin(units[tr])), np.flatnonzero(meta.isin(units[va])))
                 for tr, va in TimeSeriesSplit(args.cv, gap=args.gap).split(units)]
    classes = set(y.unique())
    for train, valid in folds:
        if args.task == "classification" and set(y.iloc[train]) != classes:
            raise ValueError("a training fold lacks a class; adjust folds or data design")
        if args.metric == "roc_auc" and y.iloc[valid].nunique() != 2:
            raise ValueError("ROC AUC requires both classes in every validation fold")
        if args.metric == "r2" and len(valid) < 2:
            raise ValueError("R2 requires at least two validation samples per fold")
    return folds


def estimator(tag, task, seed):
    return create_estimator(tag, task, seed)


def make_pipeline(model, args):
    steps = []
    if args.impute == "median":
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    if not args.no_scale:
        steps.append(("scale", StandardScaler()))
    if args.select_k:
        steps.append(("select", SelectKBest(f_classif if args.task == "classification" else f_regression,
                                             k=args.select_k)))
    if args.pca:
        steps.append(("pca", PCA(n_components=args.pca, random_state=args.seed)))
    pipeline_class = Pipeline
    if args.resample == "random-over":
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as SamplingPipeline
        pipeline_class = SamplingPipeline
        steps.append(("resample", RandomOverSampler(random_state=args.seed)))
    return pipeline_class([*steps, ("model", model)])


def evaluate(model, x, y, task):
    predicted = model.predict(x)
    frame = pd.DataFrame({"observed": y, "predicted": predicted}, index=x.index)
    if task == "regression":
        metrics = {"mae": float(mean_absolute_error(y, predicted)),
                   "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
                   "r2": float(r2_score(y, predicted)) if len(y) >= 2 and y.nunique() > 1 else None}
        frame["residual"] = y - predicted
    else:
        metrics = {"accuracy": float(accuracy_score(y, predicted)),
                   "balanced_accuracy": float(balanced_accuracy_score(y, predicted))
                   if set(y) == set(model.classes_) else None,
                   "f1_macro": float(f1_score(y, predicted, labels=model.classes_, average="macro", zero_division=0))}
        if len(model.classes_) == 2 and hasattr(model, "predict_proba"):
            frame["positive_score"] = model.predict_proba(x)[:, 1]
            frame["positive_probability"] = frame["positive_score"]
        elif len(model.classes_) == 2 and hasattr(model, "decision_function"):
            frame["positive_score"] = model.decision_function(x)
        if "positive_score" in frame and y.nunique() == 2:
            binary = y == model.classes_[1]
            metrics["roc_auc"] = float(roc_auc_score(binary, frame.positive_score))
            metrics["average_precision"] = float(average_precision_score(binary, frame.positive_score))
    return metrics, frame


def run(args):
    if not 0 < args.test_size < 1 or args.cv < 2 or args.gap < 0 or args.repeats < 2:
        raise ValueError("require 0 < test-size < 1, cv >= 2, gap >= 0 and repeats >= 2")
    if args.select_k < 0 or args.pca < 0 or args.seed < 0:
        raise ValueError("select-k, pca and seed must be nonnegative")
    if args.bootstrap != 0 and args.bootstrap < 100:
        raise ValueError("bootstrap must be 0 (disabled) or at least 100")
    if args.split != "time" and args.gap:
        raise ValueError("gap is only defined for time splitting")
    if (args.group_column and args.split != "group") or (args.time_column and args.split != "time"):
        raise ValueError("choose one split design; combined group/time splitting is not implemented")
    if args.task == "regression" and args.resample != "none":
        raise ValueError("oversampling is only supported for classification")
    args.metric = args.metric or ("balanced_accuracy" if args.task == "classification" else "mae")
    if args.metric not in SCORERS[args.task]:
        raise ValueError(f"metric must be one of {list(SCORERS[args.task])}")
    tags = list(dict.fromkeys(args.model or DEFAULT_MODELS[args.task]))
    if set(tags) - set(MODELS[args.task]):
        raise ValueError(f"unknown {args.task} models: {sorted(set(tags) - set(MODELS[args.task]))}")
    parameters = {}
    if args.model_params:
        parameters = json.loads(args.model_params.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict) or set(parameters) - set(tags):
            raise ValueError("model-params must map selected model names to parameter objects")
        if any(not isinstance(value, dict) for value in parameters.values()):
            raise ValueError("each model-params entry must be a parameter object")
    if bool(args.test_features) != bool(args.test_labels):
        raise ValueError("test-features and test-labels must be supplied together")
    if args.test_metadata and not args.test_features:
        raise ValueError("test-metadata requires external test features/labels")
    x, y = load_xy(args.features, args.labels, args.target, args.task, args.impute)
    if y.nunique() < 2:
        raise ValueError("development target needs at least two distinct values")
    if args.metric == "roc_auc" and y.nunique() != 2:
        raise ValueError("ROC AUC selection currently supports binary targets only")
    meta = load_metadata(args.metadata, x.index, args)
    if args.select_k > x.shape[1]:
        raise ValueError("select-k exceeds feature count")
    if meta is not None and meta.name in x.columns:
        raise ValueError("remove the split metadata column from predictors")
    if args.test_features:
        test_x, test_y = load_xy(args.test_features, args.test_labels, args.target, args.task, args.impute)
        if set(test_x.columns) != set(x.columns) or set(test_x.index) & set(x.index):
            raise ValueError("external test needs matching features and disjoint sample IDs")
        test_x = test_x.loc[:, x.columns]
        test_meta = load_metadata(args.test_metadata, test_x.index, args)
        if args.split == "group" and set(meta) & set(test_meta):
            raise ValueError("external test shares groups with development data")
        if args.split == "time":
            if meta.max() >= test_meta.min():
                raise ValueError("external test times must be strictly after development times")
            if args.gap:
                raise ValueError("external time test: pre-purge the boundary and use gap=0")
        train_idx, test_idx = np.arange(len(x)), None
    else:
        train_idx, test_idx = holdout_indices(x, y, meta, args)
        test_x, test_y = x.iloc[test_idx], y.iloc[test_idx]
        test_meta = meta.iloc[test_idx] if meta is not None else None
    train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
    train_meta = meta.iloc[train_idx] if meta is not None else None
    if train_y.nunique() < 2 or len(test_y) < 2:
        raise ValueError("development needs target variation; held-out test needs at least two samples")
    if args.task == "classification" and set(test_y) - set(train_y):
        raise ValueError("test contains classes absent from development data")
    folds = cv_indices(train_x, train_y, train_meta, args)
    if args.pca > min(min(len(tr) for tr, _ in folds), args.select_k or x.shape[1]):
        raise ValueError("PCA components exceed a training fold's samples or selected features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    output = args.output_dir.resolve()
    warnings = ["CV scores are used for selection, not unbiased final performance estimates.",
                "Fold SD and permutation SD are variability summaries, not confidence intervals.",
                "Permutation importance describes predictive reliance, not causal effects; correlated features can mask each other."]
    if args.split != "random":
        warnings.append("Permutation shuffles individual test rows; group/time dependence limits interpretation. No inferential p-values are computed.")
    if args.task == "classification" and test_y.nunique() != train_y.nunique():
        warnings.append("Test data does not contain every development class; some metrics and curves are undefined.")
    audit = []
    for role, frame, metadata in (("development", train_x, train_meta), ("test", test_x, test_meta)):
        for sample in frame.index:
            audit.append({"sample_id": sample, "partition": role,
                          "split_value": str(metadata.loc[sample]) if metadata is not None else ""})
    if test_idx is not None:
        for i in sorted(set(range(len(x))) - set(train_idx) - set(test_idx)):
            audit.append({"sample_id": x.index[i], "partition": "gap_excluded",
                          "split_value": str(meta.iloc[i])})
    pd.DataFrame(audit).to_csv(output / "split-membership.csv", index=False)
    fold_audit = [{"fold": number, "role": role, "sample_id": train_x.index[i]}
                  for number, (tr, va) in enumerate(folds, 1)
                  for role, indices in (("train", tr), ("validation", va)) for i in indices]
    pd.DataFrame(fold_audit).to_csv(output / "cv-membership.csv", index=False)
    records, candidates, failures, oof = [], {}, [], []
    scorer = get_scorer(SCORERS[args.task][args.metric])
    direction = -1 if args.metric in ("mae", "rmse") else 1
    with (output / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        baseline = (DummyClassifier(strategy="prior") if args.task == "classification" else
                    DummyRegressor(strategy="median" if args.metric == "mae" else "mean"))
        baseline_records = []
        for number, (tr, va) in enumerate(folds, 1):
            fitted_baseline = clone(baseline).fit(train_x.iloc[tr], train_y.iloc[tr])
            baseline_records.append({"fold": number, "score": direction * float(
                scorer(fitted_baseline, train_x.iloc[va], train_y.iloc[va]))})
        pd.DataFrame(baseline_records).to_csv(output / "baseline-cv.csv", index=False)
        for tag in tags:
            try:
                model = (create_estimator(tag, args.task, args.seed, parameters[tag]) if tag in parameters
                         else estimator(tag, args.task, args.seed))
                pipeline = make_pipeline(model, args)
                local_records, local_oof = [], []
                for number, (tr, va) in enumerate(folds, 1):
                    fitted = clone(pipeline).fit(train_x.iloc[tr], train_y.iloc[tr])
                    score = float(scorer(fitted, train_x.iloc[va], train_y.iloc[va]))
                    if not np.isfinite(score):
                        raise ValueError("non-finite validation score")
                    local_records.append({"model": tag, "fold": number, "score": direction * score,
                                          "train_score": direction * float(scorer(fitted, train_x.iloc[tr], train_y.iloc[tr])),
                                          "train_n": len(tr), "validation_n": len(va)})
                    _, fold_predictions = evaluate(fitted, train_x.iloc[va], train_y.iloc[va], args.task)
                    fold_predictions["fold"], fold_predictions["model"] = number, tag
                    local_oof.append(fold_predictions)
                candidates[tag] = pipeline
                records.extend(local_records)
                oof.extend(local_oof)
            except Exception as exc:
                failures.append({"model": tag, "error": str(exc)})
        if not candidates:
            write_json(output / "failures.json", failures)
            raise ValueError(f"all models failed; see {output / 'failures.json'}")
        cv = pd.DataFrame(records)
        cv.to_csv(output / "cv-scores.csv", index=False)
        pd.concat(oof).to_csv(output / "oof-predictions.csv", index_label="sample_id")
        comparison = [{"model": tag, "mean": float(g.score.mean()), "sd": float(g.score.std(ddof=1)),
                       "folds": len(g)} for tag, g in cv.groupby("model", sort=False)]
        comparison.sort(key=lambda r: direction * r["mean"], reverse=True)
        selected = comparison[0]["model"]
        best = candidates[selected].fit(train_x, train_y)
        metrics, predictions = evaluate(best, test_x, test_y, args.task)
        if not all(v is None or np.isfinite(v) for v in metrics.values()):
            raise ValueError("non-finite test metrics")
        predictions.to_csv(output / "test-predictions.csv", index_label="sample_id")
        baseline.fit(train_x, train_y)
        baseline_metrics, baseline_predictions = evaluate(baseline, test_x, test_y, args.task)
        baseline_predictions.to_csv(output / "baseline-predictions.csv", index_label="sample_id")
        from pipeline_analysis import feature_shift
        feature_shift(train_x, test_x).to_csv(output / "feature-shift.csv", index=False)
        train_x.corr(method="spearman").to_csv(output / "feature-correlations.csv", index_label="feature")
        importance = None
        if args.metric != "roc_auc" or test_y.nunique() == 2:
            perm = permutation_importance(best, test_x, test_y, scoring=scorer,
                                          n_repeats=args.repeats, random_state=args.seed, n_jobs=1)
            if not np.isfinite(perm.importances).all():
                raise ValueError("non-finite permutation importance")
            importance = pd.DataFrame({"feature": x.columns, "mean": perm.importances_mean,
                                       "sd": perm.importances_std}).sort_values("mean", ascending=False)
            importance.to_csv(output / "importance.csv", index=False)
            pd.DataFrame(perm.importances, index=x.columns).to_csv(output / "importance-repeats.csv", index_label="feature")
        else:
            warnings.append("ROC AUC importance omitted: test has only one class.")
        joblib.dump({"schema": "psytrainer-pipeline/v1", "pipeline": best, "features": list(x.columns),
                     "task": args.task, "target": args.target, "impute": args.impute}, output / "pipeline.joblib")
    summary = {"schema": "psytrainer-pipeline/v1", "task": args.task, "target": args.target,
               "split": args.split, "external_test": bool(args.test_features), "seed": args.seed,
               "cv_folds": args.cv, "gap": args.gap, "development_n": len(train_x), "test_n": len(test_x),
               "feature_n": x.shape[1], "metric": args.metric, "direction": "lower" if direction < 0 else "higher",
               "selected_model": selected, "comparison": comparison, "test_metrics": metrics,
               "baseline": {"strategy": baseline.strategy, "test_metrics": baseline_metrics},
               "bootstrap_repeats": args.bootstrap,
               "score_kind": ("probability" if "positive_probability" in predictions else
                              "decision" if "positive_score" in predictions else "none"),
               "classes": list(best.classes_) if args.task == "classification" else [],
               "preprocessing": {"impute": args.impute, "scale": not args.no_scale, "select_k": args.select_k,
                                 "pca": args.pca, "resample": args.resample},
               "importance_repeats": args.repeats, "failures": failures, "warnings": warnings,
               "question": args.question, "language": args.language,
               "model_backend": "local-registry/v1", "model_parameters": parameters,
               "versions": versions(),
               "inputs": {name: {"path": str(getattr(args, name).resolve()), "sha256": sha256(getattr(args, name))}
                          for name in ("features", "labels", "metadata", "test_features", "test_labels", "test_metadata", "model_params")
                          if getattr(args, name)},
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "estimator_parameters": {k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
                                        for k, v in best.named_steps["model"].get_params().items()}}
    write_json(output / "summary.json", summary)
    from pipeline_report import generate_report
    generate_report(output)
    return {"status": "completed", "selected_model": selected, "test_metrics": metrics,
            "summary": str(output / "summary.json"), "report": str(output / "report.docx"),
            "analysis": str(output / "analysis.json"),
            "figures": str(output / "figures"), "model": str(output / "pipeline.joblib"), "failures": failures}


def predict(args):
    saved = joblib.load(args.model)
    if saved.get("schema") != "psytrainer-pipeline/v1":
        raise ValueError("expected a saved Pipeline model")
    x = read_table(args.features)
    if set(x.columns) != set(saved["features"]):
        raise ValueError("prediction feature schema differs from training")
    x = x.loc[:, saved["features"]].apply(pd.to_numeric, errors="raise").astype(float)
    if x.empty or np.isinf(x).any().any() or (saved["impute"] == "none" and x.isna().any().any()):
        raise ValueError("empty or non-finite prediction data")
    values = saved["pipeline"].predict(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame({saved["target"]: values}, index=x.index).to_csv(stream, index_label="sample_id")
    return {"status": "completed", "samples": len(x), "predictions": str(args.output.resolve())}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="operation", required=True)
    t = commands.add_parser("train")
    for name in ("features", "labels", "output-dir"):
        t.add_argument(f"--{name}", type=Path, required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--task", choices=SCORERS, required=True)
    t.add_argument("--split", choices=("random", "group", "time"), default="random")
    for name in ("metadata", "test-features", "test-labels", "test-metadata"):
        t.add_argument(f"--{name}", type=Path)
    t.add_argument("--group-column")
    t.add_argument("--time-column")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--cv", type=int, default=5)
    t.add_argument("--gap", type=int, default=0, help="Purge this many distinct time blocks at split boundaries")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--model", action="append")
    t.add_argument("--model-params", type=Path, help="JSON mapping model tags to explicit estimator parameters")
    t.add_argument("--metric")
    t.add_argument("--impute", choices=("none", "median"), default="none")
    t.add_argument("--no-scale", action="store_true")
    t.add_argument("--select-k", type=int, default=0)
    t.add_argument("--pca", type=int, default=0)
    t.add_argument("--resample", choices=("none", "random-over"), default="none")
    t.add_argument("--repeats", type=int, default=10)
    t.add_argument("--bootstrap", type=int, default=500,
                   help="Test percentile bootstrap repeats, 0 to disable; time designs omit intervals")
    t.add_argument("--language", choices=("zh", "en"), default="zh")
    t.add_argument("--question", default="")
    pr = commands.add_parser("predict", help="Load only trusted local joblib files")
    for name in ("features", "model", "output"):
        pr.add_argument(f"--{name}", type=Path, required=True)
    r = commands.add_parser("report")
    r.add_argument("directory", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.operation == "train":
            value = run(args)
        elif args.operation == "predict":
            value = predict(args)
        else:
            from pipeline_report import generate_report
            generate_report(args.directory)
            value = {"report": str((args.directory / "report.docx").resolve())}
        print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
