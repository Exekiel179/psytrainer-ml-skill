# Pipeline analyses

Use the interpreter recorded in `runtime.json` (CPython 3.12-3.14). The runner
uses `scripts/model_registry.py` to construct the same 21 standard estimators
directly from their libraries, without importing or installing PsyTrainer.
It owns all splitting, preprocessing, fitting and scoring. Defaults match the
previous Pipeline. Fixed parameters use `--model-params FILE.json`; optional
bounded parameter search uses the options below. See [migration.md](migration.md).

## Commands

```bash
"$PY" scripts/pipeline_train.py train --features X.csv --labels Y.csv --target score --task regression --output-dir outputs/run-01
"$PY" scripts/pipeline_train.py train --features X.csv --labels Y.csv --target diagnosis --task classification --split group --metadata metadata.csv --group-column subject --output-dir outputs/group-01
"$PY" scripts/pipeline_train.py train --features X.csv --labels Y.csv --target score --task regression --split time --metadata metadata.csv --time-column time --gap 1 --output-dir outputs/time-01
"$PY" scripts/pipeline_train.py predict --features new.csv --model outputs/run-01/pipeline.joblib --output outputs/new-predictions.csv
"$PY" scripts/pipeline_train.py report outputs/run-01
```

Each CSV has a unique sample ID in its first column. Feature and label IDs must
match; label order is aligned automatically. Metadata is separate from predictors
and must have exactly matching IDs. Features must be numeric, with no infinity.
Targets cannot be missing. Classification supports string or numeric classes;
numeric values are treated as class labels, not an ordered outcome. One target
per run. Use fresh output directories or `--resume` for an identical interrupted
run. Only load trusted model files.
Prediction can include `--probabilities`, adding `probability::<class>` columns
when the classifier supports `predict_proba`; otherwise it fails explicitly.
Class encoding is internal, and output labels retain their original names.

## Validation and preprocessing

- Default: random 20% test holdout, reserved before 5-fold CV; classification is
  stratified, regression uses KFold. Repeatable seed 42; adjust `--seed`, `--cv`,
  `--test-size`. CV selects models; it is not nested CV or unbiased evaluation.
- Group: GroupShuffleSplit holdout and GroupKFold, with no group overlap.
  `--test-size` is a fraction of groups, not rows. No stratification guarantee.
- Time: sorted unique time blocks, forward TimeSeriesSplit and latest-block
  holdout. Equal timestamps never cross a boundary. `--gap` excludes that many
  distinct time blocks from the end of each training partition. This is not an
  elapsed-duration gap; choose a gap appropriate to feature/label horizons.
  Numeric time values or parseable date/time strings are supported.
- External test: provide `--test-features` and `--test-labels`. IDs must be
  disjoint; feature names must match. Group/time designs also require
  `--test-metadata`; groups must be disjoint or test times strictly later.
  External time tests with nonzero gap are rejected: pre-purge the boundary
  and use gap=0. Combined group-plus-time validation is not implemented.
- Preprocessing order: optional `--impute median`, standardization (disable
  with `--no-scale`), optional `--select-k N`, optional `--pca N`, optional
  classification resampling, estimator. Every learned stage is
  fitted only on training folds, then refitted on all development data.
  Missing features fail unless median imputation is explicitly selected.
  All-missing training columns are retained by the imputer using its zero
  fallback. Input preparation must not itself use validation/test information.
- Repeat `--model Model...` to compare installed estimators. Models failing a
  fold are excluded and recorded; all-model failure fails the command. The
  selected model's final fit/evaluation failure fails the command.
- Regression metrics: `mae` (default), `mse`, `rmse`, `r2`, `pearson_r`. Classification:
  `balanced_accuracy` (default), `accuracy`, macro `f1`/`precision`/`recall`, binary `roc_auc`.
  Selection uses standard sklearn scoring with proper loss direction. Displayed
  losses are positive. ROC AUC requires two classes in every validation fold.
  Positive class is the second entry in the saved sorted `classes` list.

## Migrated training options

`--preset all` compares all models for the task. Regression domain presets
`audio`, `face`, `gait`, `text` retain original domain model lists; they operate
on already extracted numeric features. They do not silently select a preprocessing
method. Explicit `--model` values override the preset model list.

`--scaler minmax` provides the original normalization family; standard scaling
remains the default and `--no-scale` disables it. `--selection f-threshold`
retains features with F >= `--f-threshold` (default 5), including the last column.
It fails if no feature survives. `--select-k N` keeps the default top-K method.

`--selection forward --select-k N` uses sequential forward selection. Omit
`--select-k` to stop when improvement falls below `--forward-tol` (default 0);
sklearn's auto mode selects at most p-1 features. No hidden PCA is applied.
Selection runs before outer preprocessing, using an internal Pipeline that fits
imputation, scaling and resampling in each inner training fold. Inner splits
use the same `--cv`, random/group/time design and gap, on the current outer
training partition only. Small groups/time series may not support this depth;
reduce the fold count or use a simpler selector, rather than dropping the design.
Selection is computationally expensive (up to O(p² * cv) fits per outer fit).

All original resampling families are available through `--resample`:
`random-over`, `smote`, `smote-tomek`, `smote-enn`, `cluster-centroids`,
`random-under`, `near-miss`. They operate only during training. Use
`--resample-params FILE.json` for library parameters, e.g. `{"k_neighbors": 3}`
for SMOTE. Defaults use the CLI seed where supported. A sampler with insufficient
minority samples fails explicitly; it is not silently replaced.

### Search and combination comparison

`--search grid` exhausts a small maintained grid per model; `--search random`
samples it reproducibly. `--max-candidates 24` limits each model; an oversized
grid fails before fitting, while random search samples up to the limit.
Provide `--search-space FILE.json` to replace a model's grid (unspecified models
use maintained grids). Parameter names can be estimator names or Pipeline names
such as `model__max_depth`. Lists of objects express conditional combinations.

```json
{
  "ModelRandomForestClassifier": [
    {"max_depth": [3, null], "preprocess__scaler": ["standard", "minmax"],
     "preprocess__resample": ["none", "random-over", "smote"]},
    {"max_depth": [3], "preprocess__selection": ["forward"],
     "preprocess__select_k": [2], "preprocess__resample": ["none"]}
  ]
}
```

`preprocess__` options are `selection`, `select_k`, `f_threshold`, `forward_tol`,
`scaler`, `no_scale`, `pca`, `resample`. Regular component parameters such as
`select__k`, `pca__n_components` and `resample__k_neighbors` are also supported.
Each candidate clones and fits the full Pipeline on shared development folds.
Forward selection has its own inner folds. Hyperparameter selection itself is
not nested CV: the best candidate's CV and OOF results are selection diagnostics,
not unbiased performance estimates. Only the selected final model is evaluated
on the held-out test. Invalid candidate configurations are recorded and excluded;
if all candidates of all models fail, the command fails.

`search-results.json` records every candidate, parameter set, mean primary score
or failure. `checkpoints/` records per-fold metrics and predictions. Reports
include search method, budget, selected parameters, failures and search figures.
`--save-candidates` also refits each successful model's best configuration on
development data and saves `candidates/<model>.joblib`, usable by `predict`.

### Resume

Repeat the same training command with `--resume` to reuse completed CV folds.
Input contents, arguments, code, Python and dependency versions must match;
changes fail before reusing results. Checkpoints are JSON, not executable pickle.
Only complete atomically written folds are reused. Baseline fits, final model
fits, diagnostics and reports run again. Keep checkpoint files intact and do
not use resume as a way to retune after viewing the test set.

## Artifacts and interpretation

The command returns one compact JSON result. Full artifacts remain on disk:

| File | Content |
|---|---|
| `summary.json` | Metrics, failures, configuration, estimator parameters, versions, input hashes |
| `pipeline.joblib` | Fitted preprocessing and selected estimator together |
| `selected-features.csv` | Final model input feature names, or PCA component names |
| `search-results.json`, `checkpoint.json`, `checkpoints/` | Candidate outcomes, run identity and completed CV folds |
| `candidates/` | Optional development-refitted best configuration per model |
| `split-membership.csv`, `cv-membership.csv` | Auditable IDs, partitions, folds; purged holdout rows are recorded |
| `cv-scores.csv` | Per-model, per-fold primary score and sample counts |
| `test-predictions.csv` | Observed/predicted values and residuals or binary scores |
| `oof-predictions.csv` | Fold validation predictions for successful candidates; selection diagnostics only |
| `baseline-cv.csv`, `baseline-predictions.csv` | Development-only fitted dummy reference, never a selection candidate |
| `metric-intervals.csv`, `analysis.json` | Test estimates, paired baseline gains, supported 95% intervals, measured findings and interpretation |
| `regression-bins.csv`, `error-cases.csv` | Prediction-level errors and all samples ordered by error |
| `class-metrics.csv`, `calibration.csv`, `thresholds.csv` | Class support, probability reliability, descriptive threshold trade-offs where defined |
| `feature-shift.csv`, `feature-correlations.csv` | Development/test mean and missingness shifts; development Spearman correlations |
| `strata-metrics.csv`, `strata-membership.csv` | Group or ordered time-block performance with counts and auditable membership |
| `importance.csv`, `importance-repeats.csv` | Test permutation importance, all features and repeats |
| `figures/` | 180 mm figures in PDF, editable SVG and 600 dpi PNG; caption/source manifest |
| `report.docx`, `report.md` | Structured methods, results, figures, limitations, provenance |
| `run.log` | Detailed model output and warnings |

Use `--language zh` (default) or `en`; `--question "..."` adds the actual research
question. Reports are deterministic templates filled from measured results,
with no LLM calls, invented references or novelty claims. Report regeneration
reads JSON/CSVs rather than loading a pickle or retraining.

Report organization adapts the question/outline/evidence/consistency principles
of academic-paper-strategist and academic-paper-composer to empirical ML.
Figure design borrows nature-figure's white backgrounds, restrained palette,
readable editable vector text, source-data traceability and explicit uncertainty.
These references inform this self-contained implementation; users do not need
those skills installed. Nature-inspired styling is not journal certification.

Figures answer separate diagnostic questions: training/validation gaps and paired
fold gains over a dummy; held-out performance and uncertainty; regression
agreement, residual spread, absolute-error CDF and error by prediction level;
classification row-normalized confusion and per-class support, ROC/PR,
probability calibration with bin counts, and threshold trade-offs; permutation
reliance, feature correlation, input shift, and group/time stability.
The report translates measured findings into bounded development-set follow-ups,
without selecting thresholds, removing features or retuning against the test set.

`--bootstrap 500` is the default; use 0 to disable or at least 100 repeats.
Random designs resample test rows; grouped designs resample entire test groups
and omit intervals below five groups. Time designs omit intervals because a
valid block length has not been specified. Intervals are 95% percentile
intervals conditional on the fitted model; they exclude training/selection
uncertainty. Model/baseline gains use the same resampled observations and are
oriented positive-is-better. A metric needs at least 100 and 80% valid repeats.
Missing-class balanced accuracy and AUC are undefined; counts stay visible.

Only genuine `predict_proba` output is calibrated; decision scores are never
treated as probabilities even when they fall in [0, 1]. Brier and ten-bin ECE
are descriptive; calibration and threshold curves have no confidence bands.
The conventional threshold line does not override the vendor prediction rule.
Fold bars are sample SD; importance bars are population SD across `--repeats`
permutations (default 10), not confidence intervals. Regression residual bands
are within-bin 10th-90th percentiles, not prediction intervals. There are no
significance tests. Top 12 importance/shift features are shown with all values
retained; every group is plotted, paginated above 20. All prediction samples
are plotted, using hexbin counts above 1500 samples. Reports from older runs
remain regenerable but cannot invent missing baseline, probability or CV data.

Permutation scores measure predictive reliance, not causality. Correlated
predictors can mask importance. Row permutation may disrupt group/time
dependence and is descriptive only. Do not tune against this test interpretation;
obtain a new untouched test set after such revisions. A held-out subset is not
automatically external, prospective or cross-site validation.
