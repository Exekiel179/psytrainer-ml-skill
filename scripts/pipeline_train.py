#!/usr/bin/env python3
"""Leakage-aware model comparison, held-out evaluation and portable prediction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from copy import copy
from pathlib import Path

if __name__ == "__main__":
    from runtime_dependencies import CORE, bootstrap
    bootstrap(__file__, CORE)

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
                             roc_auc_score, average_precision_score, get_scorer, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedKFold, TimeSeriesSplit, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from table_io import read_table
from model_registry import MODELS, create_estimator, versions
from pipeline_options import (SAMPLERS, DOMAIN_MODELS, pearson_score, FThresholdSelector, RawSequentialSelector,
                              make_sampler, search_candidates, prepare_checkpoint, atomic_json)

SCORERS = {
    "classification": {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
                       "f1": "f1_macro", "precision": "precision_macro",
                       "recall": "recall_macro", "roc_auc": "roc_auc"},
    "regression": {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                   "mse": "neg_mean_squared_error", "r2": "r2",
                   "pearson_r": make_scorer(pearson_score)},
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


def parameter_value(value):
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


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


def make_pipeline(model, args, selection_cv=None):
    steps = []
    if args.impute == "median":
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    if not args.no_scale:
        steps.append(("scale", MinMaxScaler() if args.scaler == "minmax" else StandardScaler()))
    if args.selection == "kbest" and args.select_k:
        steps.append(("select", SelectKBest(f_classif if args.task == "classification" else f_regression,
                                             k=args.select_k)))
    elif args.selection == "f-threshold":
        steps.append(("select", FThresholdSelector(
            f_classif if args.task == "classification" else f_regression, args.f_threshold)))
    if args.pca:
        steps.append(("pca", PCA(n_components=args.pca, random_state=args.seed)))
    pipeline_class = Pipeline
    if args.resample != "none":
        from imblearn.pipeline import Pipeline as SamplingPipeline
        pipeline_class = SamplingPipeline
        steps.append(("resample", make_sampler(args.resample, args.seed, args.sampler_parameters)))
    if args.selection == "forward":
        # Selection sees raw features. Its estimator fits imputation/scaling and
        # resampling anew inside inner CV, using this outer training partition only.
        inner_steps = [(name, clone(step)) for name, step in steps if name != "pca"]
        inner = pipeline_class([*inner_steps, ("model", clone(model))])
        selector = RawSequentialSelector(inner, n_features_to_select=args.select_k or "auto",
            tol=None if args.select_k else args.forward_tol, direction="forward",
            scoring=SCORERS[args.task][args.metric], cv=selection_cv, n_jobs=1)
        steps.insert(0, ("select", selector))
    return pipeline_class([*steps, ("model", model)])


def configured_pipeline(model, args, parameters, x, y, meta):
    args = candidate_options(args, parameters)
    selection_cv = cv_indices(x, y, meta, args) if args.selection == "forward" else None
    pipeline = make_pipeline(clone(model), args, selection_cv)
    parameters = {k: v for k, v in parameters.items() if not k.startswith("preprocess__")}
    pipeline.set_params(**parameters)
    if args.selection == "forward":
        # Keep the selector's estimator synchronized with the candidate's settings.
        inner = pipeline.named_steps["select"].estimator
        inner.set_params(**{k: v for k, v in parameters.items()
                            if k.split("__", 1)[0] in inner.named_steps})
    return pipeline


def candidate_options(args, parameters):
    args = copy(args)
    allowed = {"selection", "select_k", "f_threshold", "forward_tol", "scaler", "no_scale", "pca", "resample"}
    for key, value in parameters.items():
        if key.startswith("preprocess__"):
            name = key.removeprefix("preprocess__")
            if name not in allowed:
                raise ValueError(f"unsupported preprocessing option: {name}")
            setattr(args, name, value)
    if args.selection not in {"kbest", "f-threshold", "forward"} or args.scaler not in {"standard", "minmax"}:
        raise ValueError("invalid preprocessing selection or scaler")
    if args.resample not in {"none", *SAMPLERS} or (args.resample != "none" and args.task != "classification"):
        raise ValueError("resampling requires a supported classification sampler")
    if any(type(getattr(args, name)) is not int or getattr(args, name) < 0 for name in ("select_k", "pca")):
        raise ValueError("select_k and pca must be nonnegative integers")
    if type(args.no_scale) is not bool:
        raise ValueError("no_scale must be boolean")
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) < 0
           for name in ("f_threshold", "forward_tol")):
        raise ValueError("selection thresholds must be finite and nonnegative")
    if args.selection == "f-threshold" and args.select_k:
        raise ValueError("f-threshold cannot be combined with select_k")
    return args


def evaluate(model, x, y, task):
    predicted = model.predict(x)
    frame = pd.DataFrame({"observed": y, "predicted": predicted}, index=x.index)
    if task == "regression":
        metrics = {"mae": float(mean_absolute_error(y, predicted)),
                   "mse": float(mean_squared_error(y, predicted)),
                   "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
                   "r2": float(r2_score(y, predicted)) if len(y) >= 2 and y.nunique() > 1 else None}
        frame["residual"] = y - predicted
        correlation = pearson_score(y, predicted)
        metrics["pearson_r"] = correlation if np.isfinite(correlation) else None
    else:
        metrics = {"accuracy": float(accuracy_score(y, predicted)),
                   "balanced_accuracy": float(balanced_accuracy_score(y, predicted))
                   if set(y) == set(model.classes_) else None,
                   "f1_macro": float(f1_score(y, predicted, labels=model.classes_, average="macro", zero_division=0)),
                   "precision_macro": float(precision_score(y, predicted, labels=model.classes_, average="macro", zero_division=0)),
                   "recall_macro": float(recall_score(y, predicted, labels=model.classes_, average="macro", zero_division=0))}
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


def restore_labels(frame, classes):
    if classes is not None:
        frame = frame.copy()
        for column in ("observed", "predicted"):
            frame[column] = np.asarray(classes)[frame[column].to_numpy(dtype=int)]
    return frame


def run(args):
    if not 0 < args.test_size < 1 or args.cv < 2 or args.gap < 0 or args.repeats < 2:
        raise ValueError("require 0 < test-size < 1, cv >= 2, gap >= 0 and repeats >= 2")
    if args.select_k < 0 or args.pca < 0 or args.seed < 0:
        raise ValueError("select-k, pca and seed must be nonnegative")
    if args.max_candidates < 1 or not np.isfinite(args.f_threshold) or args.f_threshold < 0:
        raise ValueError("max-candidates must be positive; F threshold must be finite and nonnegative")
    if not np.isfinite(args.forward_tol) or args.forward_tol < 0:
        raise ValueError("forward-tol must be finite and nonnegative")
    if args.selection == "f-threshold" and args.select_k:
        raise ValueError("f-threshold and select-k are alternative selection rules")
    if args.search_space and args.search == "none":
        raise ValueError("search-space requires --search grid or random")
    args.sampler_parameters = json.loads(args.resample_params.read_text()) if args.resample_params else {}
    if not isinstance(args.sampler_parameters, dict) or (args.sampler_parameters and args.resample == "none"):
        raise ValueError("resample-params requires a sampler and a JSON object")
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
    if args.preset in DOMAIN_MODELS and args.task != "regression":
        raise ValueError("original domain presets define regression models only; specify classification models explicitly")
    defaults = (list(MODELS[args.task]) if args.preset == "all" else
                DOMAIN_MODELS.get(args.preset, DEFAULT_MODELS[args.task]))
    tags = list(dict.fromkeys(args.model or defaults))
    if set(tags) - set(MODELS[args.task]):
        raise ValueError(f"unknown {args.task} models: {sorted(set(tags) - set(MODELS[args.task]))}")
    parameters = {}
    if args.model_params:
        parameters = json.loads(args.model_params.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict) or set(parameters) - set(tags):
            raise ValueError("model-params must map selected model names to parameter objects")
        if any(not isinstance(value, dict) for value in parameters.values()):
            raise ValueError("each model-params entry must be a parameter object")
    grids = json.loads(args.search_space.read_text()) if args.search_space else {}
    if not isinstance(grids, dict) or set(grids) - set(tags):
        raise ValueError("search-space must map selected model names to search spaces")
    spaces = {tag: search_candidates(tag, args, grids) for tag in tags}
    if __name__ == "__main__":
        from runtime_dependencies import ensure, training_packages
        ensure(training_packages(tags, args.task, args.resample, spaces))
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
    label_classes = None
    if args.task == "classification":
        encoder = LabelEncoder().fit(train_y)
        label_classes = list(encoder.classes_)
        train_y = pd.Series(encoder.transform(train_y), index=train_y.index)
        test_y = pd.Series(encoder.transform(test_y), index=test_y.index)
    folds = cv_indices(train_x, train_y, train_meta, args)
    if args.pca > min(min(len(tr) for tr, _ in folds), args.select_k or x.shape[1]):
        raise ValueError("PCA components exceed a training fold's samples or selected features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    inputs = {name: {"path": str(getattr(args, name).resolve()), "sha256": sha256(getattr(args, name))}
              for name in ("features", "labels", "metadata", "test_features", "test_labels", "test_metadata",
                           "model_params", "search_space", "resample_params") if getattr(args, name)}
    software = versions()
    prepare_checkpoint(output, args, inputs, software)
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
    records, candidates, failures, oof, search_records = [], {}, [], [], []
    scorer = get_scorer(SCORERS[args.task][args.metric])
    direction = -1 if args.metric in ("mae", "mse", "rmse") else 1
    with (output / "run.log").open("a" if args.resume else "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        baseline = (DummyClassifier(strategy="prior") if args.task == "classification" else
                    DummyRegressor(strategy="median" if args.metric == "mae" else "mean"))
        baseline_records = []
        if args.metric == "pearson_r":
            warnings.append("Pearson correlation is undefined for a constant dummy; baseline CV correlation is omitted.")
        for number, (tr, va) in enumerate(folds, 1):
            fitted_baseline = clone(baseline).fit(train_x.iloc[tr], train_y.iloc[tr])
            baseline_records.append({"fold": number, "score": direction * float(
                scorer(fitted_baseline, train_x.iloc[va], train_y.iloc[va]))
                if args.metric != "pearson_r" else None})
        pd.DataFrame(baseline_records).to_csv(output / "baseline-cv.csv", index=False)
        for tag in tags:
            try:
                model = (create_estimator(tag, args.task, args.seed, parameters[tag]) if tag in parameters
                         else estimator(tag, args.task, args.seed))
                if tag == "ModelCatBoostRegressor":
                    model.set_params(train_dir=str(output / "catboost"))
                successful = None
                for candidate_number, candidate_params in enumerate(spaces[tag], 1):
                    local_records, local_oof = [], []
                    try:
                        for number, (tr, va) in enumerate(folds, 1):
                            checkpoint = output / "checkpoints" / f"{tag}-{candidate_number}-{number}.json"
                            if args.resume and checkpoint.is_file():
                                cached = json.loads(checkpoint.read_text())
                                record = cached["record"]
                                fold_predictions = pd.DataFrame(cached["predictions"], index=train_x.index[va])
                            else:
                                fitted = configured_pipeline(model, args, candidate_params, train_x.iloc[tr],
                                    train_y.iloc[tr], train_meta.iloc[tr] if train_meta is not None else None)
                                fitted.fit(train_x.iloc[tr], train_y.iloc[tr])
                                score = float(scorer(fitted, train_x.iloc[va], train_y.iloc[va]))
                                train_score = float(scorer(fitted, train_x.iloc[tr], train_y.iloc[tr]))
                                if not np.isfinite([score, train_score]).all():
                                    raise ValueError("non-finite CV score")
                                record = {"model": tag, "fold": number, "score": direction * score,
                                          "train_score": direction * train_score,
                                          "train_n": len(tr), "validation_n": len(va)}
                                fold_metrics, fold_predictions = evaluate(fitted, train_x.iloc[va], train_y.iloc[va], args.task)
                                record.update({f"validation_{key}": value for key, value in fold_metrics.items()})
                                fold_predictions["fold"], fold_predictions["model"] = number, tag
                                inner_audit = []
                                selector = fitted.named_steps.get("select")
                                if isinstance(selector, RawSequentialSelector):
                                    inner_audit = [{"fold": fold, "train_ids": list(train_x.index[tr][itr]),
                                                    "validation_ids": list(train_x.index[tr][iva])}
                                                   for fold, (itr, iva) in enumerate(selector.cv, 1)]
                                atomic_json(checkpoint, {"record": record, "inner_cv": inner_audit,
                                    "predictions": fold_predictions.to_dict(orient="list")})
                            local_records.append(record)
                            local_oof.append(fold_predictions)
                        mean = float(np.mean([r["score"] for r in local_records]))
                        if successful is None or direction * mean > successful[0]:
                            successful = (direction * mean, candidate_params, local_records, local_oof)
                        search_records.append({"model": tag, "candidate": candidate_number,
                            "parameters": candidate_params, "mean": mean, "status": "completed"})
                    except Exception as exc:
                        search_records.append({"model": tag, "candidate": candidate_number,
                            "parameters": candidate_params, "status": "failed", "error": str(exc)})
                    atomic_json(output / "search-results.json", search_records)
                if successful is None:
                    raise ValueError("all parameter candidates failed; see search-results.json")
                _, chosen_params, local_records, local_oof = successful
                candidates[tag] = (model, chosen_params)
                records.extend(local_records)
                oof.extend(local_oof)
            except Exception as exc:
                failures.append({"model": tag, "error": str(exc)})
        if not candidates:
            write_json(output / "failures.json", failures)
            raise ValueError(f"all models failed; see {output / 'failures.json'}")
        cv = pd.DataFrame(records)
        cv.to_csv(output / "cv-scores.csv", index=False)
        restore_labels(pd.concat(oof), label_classes).to_csv(output / "oof-predictions.csv", index_label="sample_id")
        comparison = [{"model": tag, "mean": float(g.score.mean()), "sd": float(g.score.std(ddof=1)),
                       "folds": len(g)} for tag, g in cv.groupby("model", sort=False)]
        comparison.sort(key=lambda r: direction * r["mean"], reverse=True)
        selected = comparison[0]["model"]
        model, selected_parameters = candidates[selected]
        best = configured_pipeline(model, args, selected_parameters, train_x, train_y, train_meta).fit(train_x, train_y)
        feature_names = np.asarray(x.columns, dtype=object)
        for name, step in best.steps[:-1]:
            if hasattr(step, "get_feature_names_out"):
                feature_names = step.get_feature_names_out(feature_names)
        pd.DataFrame({"feature": feature_names}).to_csv(output / "selected-features.csv", index=False)
        if args.save_candidates:
            candidate_dir = output / "candidates"
            candidate_dir.mkdir(exist_ok=True)
            for tag, (candidate_model, candidate_params) in candidates.items():
                fitted_candidate = best if tag == selected else configured_pipeline(
                    candidate_model, args, candidate_params, train_x, train_y, train_meta).fit(train_x, train_y)
                joblib.dump({"schema": "psytrainer-pipeline/v1", "pipeline": fitted_candidate,
                    "features": list(x.columns), "task": args.task, "target": args.target,
                    "impute": args.impute, "label_classes": label_classes}, candidate_dir / f"{tag}.joblib")
        metrics, predictions = evaluate(best, test_x, test_y, args.task)
        predictions = restore_labels(predictions, label_classes)
        if not all(v is None or np.isfinite(v) for v in metrics.values()):
            raise ValueError("non-finite test metrics")
        predictions.to_csv(output / "test-predictions.csv", index_label="sample_id")
        baseline.fit(train_x, train_y)
        baseline_metrics, baseline_predictions = evaluate(baseline, test_x, test_y, args.task)
        baseline_predictions = restore_labels(baseline_predictions, label_classes)
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
                     "task": args.task, "target": args.target, "impute": args.impute,
                     "label_classes": label_classes}, output / "pipeline.joblib")
    chosen_options = candidate_options(args, selected_parameters)
    summary = {"schema": "psytrainer-pipeline/v1", "task": args.task, "target": args.target,
               "split": args.split, "external_test": bool(args.test_features), "seed": args.seed,
               "cv_folds": args.cv, "gap": args.gap, "development_n": len(train_x), "test_n": len(test_x),
               "feature_n": x.shape[1], "metric": args.metric, "direction": "lower" if direction < 0 else "higher",
               "selected_model": selected, "comparison": comparison, "test_metrics": metrics,
               "baseline": {"strategy": baseline.strategy, "test_metrics": baseline_metrics},
               "bootstrap_repeats": args.bootstrap,
               "score_kind": ("probability" if "positive_probability" in predictions else
                              "decision" if "positive_score" in predictions else "none"),
               "classes": label_classes or [],
               "class_labels": label_classes,
               "search": {"method": args.search, "max_candidates": args.max_candidates,
                          "evaluated_candidates": len(search_records), "selected_parameters": selected_parameters,
                          "failed_candidates": sum(r["status"] == "failed" for r in search_records)},
               "fitted_preprocessing_parameters": {name: {k: parameter_value(v)
                   for k, v in step.get_params(deep=False).items() if k not in {"estimator", "cv", "scoring"}}
                   for name, step in best.steps[:-1]},
               "preprocessing": {"impute": args.impute, "scale": not chosen_options.no_scale,
                                 **{key: getattr(chosen_options, key) for key in
                                    ("select_k", "scaler", "selection", "f_threshold", "forward_tol", "pca", "resample")}},
               "importance_repeats": args.repeats, "failures": failures, "warnings": warnings,
               "question": args.question, "language": args.language,
               "model_backend": "local-registry/v1", "model_parameters": parameters,
               "versions": software,
               "inputs": inputs,
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "estimator_parameters": {k: parameter_value(v)
                                        for k, v in best.named_steps["model"].get_params().items()}}
    write_json(output / "summary.json", summary)
    from pipeline_report import generate_report
    generate_report(output)
    return {"status": "completed", "selected_model": selected, "test_metrics": metrics,
            "summary": str(output / "summary.json"), "report": str(output / "report.docx"),
            "analysis": str(output / "analysis.json"),
            "figures": str(output / "figures"), "model": str(output / "pipeline.joblib"), "failures": failures}


def predict(args):
    from runtime_dependencies import load_model
    saved = load_model(args.model)
    if saved.get("schema") != "psytrainer-pipeline/v1":
        raise ValueError("expected a saved Pipeline model")
    x = read_table(args.features)
    if set(x.columns) != set(saved["features"]):
        raise ValueError("prediction feature schema differs from training")
    x = x.loc[:, saved["features"]].apply(pd.to_numeric, errors="raise").astype(float)
    if x.empty or np.isinf(x).any().any() or (saved["impute"] == "none" and x.isna().any().any()):
        raise ValueError("empty or non-finite prediction data")
    if args.probabilities and saved["task"] != "classification":
        raise ValueError("probabilities require a classification model")
    values = saved["pipeline"].predict(x)
    if saved.get("label_classes") is not None:
        values = np.asarray(saved["label_classes"])[np.asarray(values, dtype=int)]
    frame = pd.DataFrame({saved["target"]: values}, index=x.index)
    if args.probabilities:
        if not hasattr(saved["pipeline"], "predict_proba"):
            raise ValueError("this classifier has no predict_proba; decision scores are not probabilities")
        classes = saved.get("label_classes") or list(saved["pipeline"].classes_)
        probabilities = saved["pipeline"].predict_proba(x)
        for i, label in enumerate(classes):
            column = f"probability::{label}"
            if column in frame:
                raise ValueError("target name conflicts with probability output column")
            frame[column] = probabilities[:, i]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index_label="sample_id")
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
    t.add_argument("--preset", choices=("default", "all", *DOMAIN_MODELS), default="default")
    t.add_argument("--save-candidates", action="store_true", help="Save each successful model refitted on development data")
    t.add_argument("--model-params", type=Path, help="JSON mapping model tags to explicit estimator parameters")
    t.add_argument("--search", choices=("none", "grid", "random"), default="none")
    t.add_argument("--search-space", type=Path, help="JSON model tags to parameter value lists (or conditional grids)")
    t.add_argument("--max-candidates", type=int, default=24, help="Per-model search budget")
    t.add_argument("--resume", action="store_true", help="Reuse completed CV folds of an identical local run")
    t.add_argument("--metric")
    t.add_argument("--impute", choices=("none", "median"), default="none")
    t.add_argument("--no-scale", action="store_true")
    t.add_argument("--scaler", choices=("standard", "minmax"), default="standard")
    t.add_argument("--selection", choices=("kbest", "f-threshold", "forward"), default="kbest")
    t.add_argument("--f-threshold", type=float, default=5.0)
    t.add_argument("--forward-tol", type=float, default=0.0)
    t.add_argument("--select-k", type=int, default=0)
    t.add_argument("--pca", type=int, default=0)
    t.add_argument("--resample", choices=("none", *SAMPLERS), default="none")
    t.add_argument("--resample-params", type=Path, help="JSON sampler parameters, e.g. k_neighbors for SMOTE")
    t.add_argument("--repeats", type=int, default=10)
    t.add_argument("--bootstrap", type=int, default=500,
                   help="Test percentile bootstrap repeats, 0 to disable; time designs omit intervals")
    t.add_argument("--language", choices=("zh", "en"), default="zh")
    t.add_argument("--question", default="")
    pr = commands.add_parser("predict", help="Load only trusted local joblib files")
    pr.add_argument("--probabilities", action="store_true", help="Include class probabilities when supported")
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
            if __name__ == "__main__":
                from runtime_dependencies import REPORT, ensure
                ensure(REPORT)
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
