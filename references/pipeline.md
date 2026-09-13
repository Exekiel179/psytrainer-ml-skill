# Pipeline analyses

Use the interpreter recorded in `runtime.json`. The new runner extracts an
unfitted sklearn-compatible estimator from the bundled wheel's model registry.
It owns all splitting, preprocessing, fitting and scoring; it does not call the
vendor Trainer, its pre-CV filters, resamplers or scorer implementation.
Default estimator parameters are used, without the vendor's parameter grids.

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
per run. Use fresh output directories. Only load trusted model files.

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
  classification `--resample random-over`, estimator. Every learned stage is
  fitted only on training folds, then refitted on all development data.
  Missing features fail unless median imputation is explicitly selected.
  All-missing training columns are retained by the imputer using its zero
  fallback. Input preparation must not itself use validation/test information.
- Repeat `--model Model...` to compare installed estimators. Models failing a
  fold are excluded and recorded; all-model failure fails the command. The
  selected model's final fit/evaluation failure fails the command.
- Regression metrics: `mae` (default), `rmse`, `r2`. Classification:
  `balanced_accuracy` (default), `accuracy`, macro `f1`, binary `roc_auc`.
  Selection uses standard sklearn scoring with proper loss direction. Displayed
  losses are positive. ROC AUC requires two classes in every validation fold.
  Positive class is the second entry in the saved sorted `classes` list.

## Artifacts and interpretation

The command returns one compact JSON result. Full artifacts remain on disk:

| File | Content |
|---|---|
| `summary.json` | Metrics, failures, configuration, estimator parameters, versions, input hashes |
| `pipeline.joblib` | Fitted preprocessing and selected estimator together |
| `split-membership.csv`, `cv-membership.csv` | Auditable IDs, partitions, folds; purged holdout rows are recorded |
| `cv-scores.csv` | Per-model, per-fold primary score and sample counts |
| `test-predictions.csv` | Observed/predicted values and residuals or binary scores |
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

Plots: fold comparison, regression observed-vs-predicted and residuals, or
classification confusion matrix plus binary ROC/precision-recall where defined;
permutation importance uses original input features through the full Pipeline.
Fold bars are sample SD; importance bars are population SD across `--repeats`
permutations (default 10). Neither is a confidence interval. There are no
significance tests. Only the top 15 importance features are plotted; all values
are retained. No samples are subsampled for prediction scatterplots.

Permutation scores measure predictive reliance, not causality. Correlated
predictors can mask importance. Row permutation may disrupt group/time
dependence and is descriptive only. Do not tune against this test interpretation;
obtain a new untouched test set after such revisions. A held-out subset is not
automatically external, prospective or cross-site validation.
