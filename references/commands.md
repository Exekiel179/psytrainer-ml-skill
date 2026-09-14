# Commands

This reference describes the original INI engine. Install with
`scripts/install_runtime.py --legacy` and use `runtime-legacy.json` -> `python`
as `$PY`. For the independent Pipeline use [pipeline.md](pipeline.md).
`ml.py capabilities` lists Pipeline models; `ml.py capabilities --legacy` lists
the original engine's models and metrics. Commands print compact JSON;
read only files referenced by the response when more detail is needed.

## Inspect and Configure

```bash
"$PY" scripts/ml.py inspect --features features.csv --labels labels.csv --task regression
"$PY" scripts/ml.py configure --features features.csv --labels labels.csv --task regression --target score --config config/run.ini --output-dir outputs/run-01
```

`configure` includes inspection, so skip a separate `inspect` when task, target,
and paths are already known. No data is silently cleaned or dropped. CSV first
columns are unique sample IDs, preserved as strings including leading zeros.
Feature values must be finite numbers. Classification presets require binary
0/1 labels because the vendor uses binary scorers. Ask before recoding labels.

Defaults: `--preset quick` selects two models; `--preset standard` selects three.
Use repeatable `--model TAG` for explicit algorithms and `--metric NAME` for the
primary selection metric; get supported values from `capabilities --legacy` only when needed.
`--cv 5` controls fold count; repeat `--target` to select multiple label columns.
Generated configs remember target selection and use no preprocessing, no
resampling, no parameter grid search, and no deletion of existing results.
`estimated_fits` is an estimate of CV plus final fits, not a runtime guarantee.

## Execute and Summarize

```bash
"$PY" scripts/ml.py train --config config/run.ini
"$PY" scripts/ml.py predict --config config/predict.ini
"$PY" scripts/ml.py report outputs/run-01/training-summary.json --top 3
```

Training/prediction validate inputs, refuse nonempty output directories, send
all subprocess output to `run.log`, and write `run-manifest.json` with input/config
hashes, command, interpreter and exit code. Use `--output-dir` for a new run.
Training accepts repeatable `--target`; prediction accepts repeatable `--model`.
JSON includes full summary paths and at most `--top` (1-20) models per target.
Full results remain in `training-summary.json` / `prediction-summary.json`.
On failure inspect the end of `run.log`; a failed run is not a successful result.

Prediction INI uses `[batch_predict]` with `feature_file`, `model_root` and
`output_dir`; paths are relative to the config. Model directories usually contain
`model.pkl` and `feature_filter.csv`, optionally `pca/pca.m`. For vendor unfiltered
`NoneType&...` models, the saved estimator's feature names are used in order when
the filter file is absent. Missing names or missing required features fail.
These commands do not establish trust in a pickle: use trusted local models only.

Legacy `PsyTrainer.py` / `batch_predict.py` remain available for verbose output,
data-only `--dry-run`, and existing callers. Their output directories and overwrite
behavior follow the INI; prefer `ml.py` for new runs.

## Interpretation Limits

The installed wheel offers 9 classification and 12 regression model wrappers,
4 feature filters, and 7 resamplers. These are classical tabular algorithms;
text/audio/face/gait presets are not raw-media encoders or deep-learning pipelines.

Vendor filtering and resampling occur before cross-validation, and folds are
random KFold rather than grouped, temporal or stratified splits. Feature selection
and resampling can leak validation information. Even binary data with enough
minority examples can produce a single-class fold. Random seeds are not fully
controlled. Do not represent results as confirmatory or held-out performance.

Rank only within the same target, data and primary metric. Vendor loss scores
are negated (higher is better). The bundled vendor also negates R2 despite larger
R2 being better; `ml.py` refuses R2 as the primary selection metric. Secondary R2
in legacy reports also has reversed sign. No confidence interval is synthesized.
Grouped/longitudinal validation, fold-contained preprocessing, multiclass scoring
and external holdout evaluation are provided by the independent Pipeline.
SHAP and deep learning are not implemented by either runner.
