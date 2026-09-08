---
name: psytrainer-ml
description: >-
  Configure and run PsyTrainer tabular model training and batch prediction from
  natural-language requests. Use when the user provides feature/label CSVs, asks
  to run PsyTrainer, compare its model results, or predict with saved model
  directories. Prefer explicit invocation (/skill:psytrainer-ml) over ambient guesses.
metadata:
  short-description: Train and predict with PsyTrainer
---

# PsyTrainer ML

Turn the user's natural-language request into a PsyTrainer training or prediction run. Use only the two bundled programs for execution:

- `scripts/PsyTrainer.py` trains one or more target columns.
- `scripts/batch_predict.py` applies saved model directories to a feature CSV.

Prefer explicit `/skill:psytrainer-ml` (or an equivalent named invoke). Do not treat every mention of “train a model” as automatic activation when another analysis path is already in use.

## Workflow

1. Determine whether the user wants training, prediction, or interpretation of existing outputs.
2. Inspect the CSV headers and sample IDs. Ask only for choices that cannot be inferred, such as classification versus regression, target columns, or the intended scoring metric.
3. Create or update `config/ml.ini` using `config/ml.ini.example`. Paths may be absolute or relative to the config file. For a smoke check, copy `fixtures/*.csv` paths from the example comments.
4. Detect a usable Python that can import `ccpl_training_models` (often 3.13 with the PsyTrainer wheel). Prefer that interpreter; do not install packages unless the user asks. If unavailable, state the missing runtime clearly and stop.
5. Run `--dry-run` first to validate paths, table shape, sample alignment, target/model selection, and output location.
6. Run the requested operation.
7. Read `training-summary.json` or `prediction-summary.json` and the framework's result files before explaining outcomes.

## Training

```bash
python3 scripts/PsyTrainer.py --config config/ml.ini --dry-run
python3 scripts/PsyTrainer.py --config config/ml.ini
```

Replace `python3` with the interpreter that has PsyTrainer installed when it is not the default. Add `--target NAME` one or more times to limit training to selected label columns. The script aligns labels to features by the first CSV column, rejects duplicate or mismatched sample IDs, and rejects non-numeric or non-finite features. It creates one output directory per target and writes `training-summary.json`.

Known configuration keywords from the supplied program are `all_ff` for all feature filters, `regression` or `classification` for model families, and `all_resample` for all resampling methods. Do not invent other framework-specific tags; use values documented by the user's PsyTrainer package.

## Prediction

```bash
python3 scripts/batch_predict.py --config config/ml.ini --dry-run
python3 scripts/batch_predict.py --config config/ml.ini
```

Add `--model NAME` one or more times to select model directories. Each immediate model directory must contain `model.pkl` and `feature_filter.csv`; `pca/pca.m` is optional.

**Trust boundary:** `model.pkl` is loaded with joblib/pickle and can execute code. Only predict from model directories written by a trusted local training run in this project, or paths the user explicitly marks as trusted. Never download and load an untrusted pickle.

The script preserves every input row, writes one CSV per successful model, includes probabilities when `predict_proba` is available, and records failures in `prediction-summary.json`.

## Interpret Results

- Distinguish exploratory model comparison from confirmatory analysis.
- Report the sample count, target/model, validation setup, metric definitions, and failed models.
- Do not infer causality from prediction or feature selection.
- Do not invent confidence intervals or performance values that PsyTrainer did not output.
- Flag leakage risks, inappropriate random cross-validation for grouped or longitudinal data, class imbalance, and reuse of training data as test data.
