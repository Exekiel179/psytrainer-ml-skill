"""Locally maintained preprocessing, bounded search and resumable CV artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.model_selection import ParameterGrid, ParameterSampler


SAMPLERS = {
    "random-over": ("over_sampling", "RandomOverSampler"),
    "smote": ("over_sampling", "SMOTE"),
    "smote-tomek": ("combine", "SMOTETomek"),
    "smote-enn": ("combine", "SMOTEENN"),
    "cluster-centroids": ("under_sampling", "ClusterCentroids"),
    "random-under": ("under_sampling", "RandomUnderSampler"),
    "near-miss": ("under_sampling", "NearMiss"),
}

DOMAIN_MODELS = {
    "audio": ["ModelRandomForestRegressor", "ModelKNeighborsRegressor"],
    "face": ["ModelRandomForestRegressor", "ModelCatBoostRegressor", "ModelSVRRegressor",
             "ModelXGBoostRegressor", "ModelLGBMRegressor"],
    "gait": ["ModelRandomForestRegressor", "ModelSVRRegressor", "ModelGPRRegressor", "ModelLRRegressor"],
    "text": ["ModelExtraTreeRegressor", "ModelAdaBoostRegressor", "ModelGradientBoostingRegressor",
             "ModelBaggingRegressor", "ModelKNeighborsRegressor", "ModelRandomForestRegressor"],
}


def pearson_score(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(predicted) == 0:
        return np.nan
    return float(np.corrcoef(y, predicted)[0, 1])


class FThresholdSelector(SelectKBest):
    """Retain every feature meeting the F threshold, including the last column."""

    def __init__(self, score_func, threshold=5.0):
        super().__init__(score_func=score_func, k="all")
        self.threshold = threshold

    def _get_support_mask(self):
        support = ~np.isnan(self.scores_) & (self.scores_ >= self.threshold)
        if not support.any():
            raise ValueError("F threshold retained no features; lower it using development data")
        return support


class RawSequentialSelector(SequentialFeatureSelector):
    """Let the internal estimator Pipeline impute raw data inside selection CV."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _get_best_new_feature_score(self, *args, **kwargs):
        index, score = super()._get_best_new_feature_score(*args, **kwargs)
        if not np.isfinite(score):
            raise ValueError("forward selection produced an undefined inner CV score")
        return index, score


def make_sampler(name, seed, parameters=None):
    module, class_name = SAMPLERS[name]
    sampler = getattr(import_module(f"imblearn.{module}"), class_name)()
    params = sampler.get_params()
    if "random_state" in params:
        sampler.set_params(random_state=seed)
    if parameters:
        sampler.set_params(**parameters)
    return sampler


def compact_grid(tag):
    """Small, valid starting grids; explicit JSON can replace these completely."""
    if "KNeighbors" in tag:
        return {"model__n_neighbors": [3, 5], "model__weights": ["uniform", "distance"]}
    if tag == "ModelLRClassifier":
        return {"model__C": [0.1, 1.0, 10.0]}
    if tag == "ModelLRRegressor":
        return {"model__fit_intercept": [True, False], "model__positive": [False, True]}
    if "SVC" in tag or "SVR" in tag:
        return {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}
    if "NaiveBayes" in tag:
        return {"model__var_smoothing": [1e-9, 1e-7, 1e-5]}
    if "GPR" in tag:
        return {"model__alpha": [1e-10, 1e-6, 0.01], "model__normalize_y": [False, True]}
    if "CatBoost" in tag:
        return {"model__iterations": [50, 100], "model__depth": [3, 6]}
    if "LGBM" in tag:
        return {"model__n_estimators": [50, 100], "model__num_leaves": [7, 15]}
    if "XGBoost" in tag or "GradientBoosting" in tag:
        return {"model__n_estimators": [50, 100], "model__max_depth": [2, 4]}
    if "AdaBoost" in tag:
        return {"model__n_estimators": [25, 50], "model__learning_rate": [0.1, 1.0]}
    if "Bagging" in tag:
        return {"model__n_estimators": [10, 30], "model__max_samples": [0.5, 1.0]}
    return {"model__max_depth": [3, None], "model__min_samples_leaf": [1, 3]}


def search_candidates(tag, args, grids):
    if args.search == "none":
        return [{}]
    raw = grids[tag] if tag in grids else compact_grid(tag)
    parts = raw if isinstance(raw, list) else [raw]
    normalized = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("each search space must be an object or list of objects")
        grid = {}
        for key, values in part.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"search values for {key} must be a nonempty list")
            key = key if "__" in key else f"model__{key}"
            if key.split("__", 1)[0] not in {"model", "scale", "impute", "select", "pca", "resample", "preprocess"}:
                raise ValueError(f"unsupported search step: {key}")
            if key in grid:
                raise ValueError(f"duplicate normalized search parameter: {key}")
            if key.startswith("select__") and key not in {
                "select__k", "select__threshold", "select__n_features_to_select", "select__tol"
            }:
                raise ValueError("search cannot override selector CV, scoring or estimator")
            grid[key] = values
        normalized.append(grid)
    space = ParameterGrid(normalized)
    if args.search == "grid":
        if len(space) > args.max_candidates:
            raise ValueError(f"{tag}: {len(space)} candidates exceed --max-candidates {args.max_candidates}")
        return list(space)
    return list(ParameterSampler(normalized, n_iter=min(args.max_candidates, len(space)),
                                 random_state=args.seed))


def prepare_checkpoint(output, args, inputs, software):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "output_dir"}}
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob("*.py")}
    identity = {"arguments": config, "inputs": inputs, "versions": software, "sources": sources,
                "python": sys.version}
    path = output / "checkpoint.json"
    if args.resume:
        if not path.is_file() or json.loads(path.read_text()) != identity:
            raise ValueError("resume requires identical arguments, input contents, code and library versions")
    else:
        if any(output.iterdir()):
            raise ValueError("output directory must be empty; use --resume for an unchanged interrupted run")
        atomic_json(path, identity)
    (output / "checkpoints").mkdir(exist_ok=True)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
