"""Local model names compatible with PsyTrainer 0.2.0's estimator mapping.

The mapping was checked against CCPL's Apache-2.0 distribution (see NOTICE.md).
Estimators use library defaults plus the existing Pipeline's execution settings;
vendor search grids and pre-CV processing are not applied here.
"""

from __future__ import annotations

from importlib import import_module, metadata


MODELS = {
    "classification": {
        "ModelAdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier"),
        "ModelDecisionTreesClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
        "ModelKNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier"),
        "ModelLGBMClassifier": ("lightgbm.sklearn", "LGBMClassifier"),
        "ModelLRClassifier": ("sklearn.linear_model", "LogisticRegression"),
        "ModelNaiveBayesClassifier": ("sklearn.naive_bayes", "GaussianNB"),
        "ModelRandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
        "ModelSVCClassifier": ("sklearn.svm", "SVC"),
        "ModelXGBoostClassifier": ("xgboost", "XGBClassifier"),
    },
    "regression": {
        "ModelAdaBoostRegressor": ("sklearn.ensemble", "AdaBoostRegressor"),
        "ModelBaggingRegressor": ("sklearn.ensemble", "BaggingRegressor"),
        "ModelCatBoostRegressor": ("catboost", "CatBoostRegressor"),
        "ModelExtraTreeRegressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
        "ModelGPRRegressor": ("sklearn.gaussian_process", "GaussianProcessRegressor"),
        "ModelGradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
        "ModelKNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor"),
        "ModelLGBMRegressor": ("lightgbm.sklearn", "LGBMRegressor"),
        "ModelLRRegressor": ("sklearn.linear_model", "LinearRegression"),
        "ModelRandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
        "ModelSVRRegressor": ("sklearn.svm", "SVR"),
        "ModelXGBoostRegressor": ("xgboost", "XGBRegressor"),
    },
}


def create_estimator(tag, task, seed, parameters=None):
    if tag not in MODELS.get(task, {}):
        raise ValueError(f"unknown {task} model: {tag}")
    module, name = MODELS[task][tag]
    try:
        model = getattr(import_module(module), name)()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{tag} requires a working {module.split('.')[0]} installation; "
            "run scripts/install_runtime.py --packages "
            f"{'scikit-learn' if module.startswith('sklearn.') else module.split('.')[0]} "
            "and check any reported system-library errors"
        ) from exc
    params = model.get_params(deep=False)
    updates = {k: seed for k in ("random_state", "random_seed") if k in params}
    if "n_jobs" in params:
        updates["n_jobs"] = 1
    if tag == "ModelLRClassifier":
        updates["max_iter"] = 2000
    if parameters:
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters for {tag} must be a JSON object")
        # CatBoost exposes only explicitly set constructor parameters.
        if module != "catboost" and set(parameters) - set(params):
            raise ValueError(f"unknown parameters for {tag}: {sorted(set(parameters) - set(params))}")
        updates.update(parameters)
    model.set_params(**updates)
    return model


def versions():
    result = {}
    for name in ("scikit-learn", "lightgbm", "xgboost", "catboost", "imbalanced-learn",
                 "numpy", "scipy", "pandas", "joblib", "matplotlib", "python-docx"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result
