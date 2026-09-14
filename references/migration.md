# Model migration and compatibility

The Skill preserves the workflow: align CSVs by sample ID, select algorithms and
parameters, compare candidates, refit, save models, predict, and interpret results.
The independent Pipeline's validation, fold-local preprocessing and report
implementation are retained. Estimator construction lives in
`scripts/model_registry.py`; preprocessing, search and resume now live in the
local Pipeline. The itemized [wheel audit](wheel-audit.md) records migration
decisions, intentionally retired behavior and validation evidence.

## Model mapping

All 21 public tags are retained. `ModelExtraTreeRegressor` intentionally maps to
**ExtraTreesRegressor**, as in the original package (not a single ExtraTreeRegressor).

| Tags (prefix `Model`) | Underlying estimators |
|---|---|
| AdaBoostClassifier / AdaBoostRegressor | sklearn AdaBoostClassifier / AdaBoostRegressor |
| DecisionTreesClassifier | sklearn DecisionTreeClassifier |
| KNeighborsClassifier / KNeighborsRegressor | sklearn KNeighborsClassifier / KNeighborsRegressor |
| LRClassifier / LRRegressor | sklearn LogisticRegression / LinearRegression |
| NaiveBayesClassifier | sklearn GaussianNB |
| RandomForestClassifier / RandomForestRegressor | sklearn RandomForestClassifier / RandomForestRegressor |
| SVCClassifier / SVRRegressor | sklearn SVC / SVR |
| BaggingRegressor | sklearn BaggingRegressor |
| ExtraTreeRegressor | sklearn ExtraTreesRegressor |
| GPRRegressor | sklearn GaussianProcessRegressor |
| GradientBoostingRegressor | sklearn GradientBoostingRegressor |
| LGBMClassifier / LGBMRegressor | LightGBM LGBMClassifier / LGBMRegressor |
| XGBoostClassifier / XGBoostRegressor | XGBoost XGBClassifier / XGBRegressor |
| CatBoostRegressor | CatBoost CatBoostRegressor |

Defaults are those of the installed algorithm library. As in the previous
Pipeline, exposed `random_state`/`random_seed` parameters use `--seed`, exposed
`n_jobs` uses 1, and LogisticRegression uses `max_iter=2000`. SVC does not enable
probabilities automatically. CatBoost's empty default parameter dictionary does
not expose seed/thread keys, so its defaults remain unchanged. Dependency
versions are recorded; equal behavior is checked within the same environment,
not promised across different library versions or operating systems.

## Explicit parameters

Create a JSON file mapping selected tags to estimator keyword arguments:

```json
{
  "ModelRandomForestRegressor": {"n_estimators": 300, "max_depth": 6},
  "ModelLRRegressor": {"fit_intercept": true}
}
```

Pass `--model-params config/models.json` to `pipeline_train.py train`, with
`--model` arguments if the desired models differ from the defaults. Only selected
models may have entries. Parameters override defaults, including seed/thread
settings when explicitly supplied. They are applied before fold cloning and
recorded with the file hash and fitted estimator parameters in `summary.json`.
Invalid model names or JSON structure fail; invalid estimator settings are
reported as model failures, and an all-model failure stops training.

These are fixed estimator settings. Parameter search uses `--search grid|random`
and optional `--search-space FILE.json`; without a file it uses small maintained
grids. All four original preprocessing families and seven resamplers have local
equivalents. Domain model lists use `--preset audio|face|gait|text`; preprocessing
is explicit. See [Pipeline options](pipeline.md#migrated-training-options).
The old exhaustive grids are intentionally not copied: they contain invalid
combinations and overly large searches. Conditional JSON spaces preserve custom
search capability. Numerical equivalence to the old pre-CV pipeline is not claimed.

## Installation and existing work

Default: `python scripts/install_runtime.py`, CPython 3.12-3.14, `.venv`,
`runtime.json`. All 21 model dependencies and report libraries are installed.
Neither source imports nor provenance collection requires PsyTrainer.

Original INI compatibility only: `python scripts/install_runtime.py --legacy --wheel /path/to/PsyTrainer.whl`,
with an externally supplied package and matching CPython, `.venv-legacy`,
`runtime-legacy.json`. Existing INI files and model directories still use
`ml.py configure/train/predict/report`, `PsyTrainer.py` and `batch_predict.py`.
The two environments coexist; neither installation deletes the other's marker.
`--recreate` replaces only the selected mode's environment.

Pre-migration `.venv` installations containing PsyTrainer continue to work; no
automatic uninstall or INI conversion occurs. Reinstall to refresh dependencies
and runtime metadata. Do not replace original INI execution with Pipeline and
claim identical scores: CV design, preprocessing and scoring differ intentionally.

Saved Pipeline v1 joblib dictionaries retain the same schema and standard-library
estimator classes, so the migration introduces no custom pickle class dependency.
Newly saved F-threshold/forward Pipelines also require the local
`scripts/pipeline_options.py` module; use this Skill's prediction CLI.
Original `model.pkl`, feature selections and PCA files continue through the
original prediction CLI. Keep the original Python and library versions when
loading historical models; cross-version pickle compatibility is not guaranteed.
Load only trusted model files. Reports can be regenerated from JSON/CSV without
loading the fitted model.

## Validation

Registry tests fit/predict and serialize every mapped estimator without vendor
imports. Where the optional engine is installed, tests compare every model class
and parameter dictionary against the original factory. Pipeline tests retain
fold-local imputation, split separation, held-out selection isolation and saved
prediction checks. `tests/smoke_pipeline.py` verifies random/group/time workflows,
diagnostics, figures and Word output; `smoke_runtime.py` and `smoke_agent.py` verify
the original engine. CI covers Pipeline on Python 3.12/3.13/3.14 across Linux,
Windows and macOS, and the original engine on Python 3.14 across those systems.
