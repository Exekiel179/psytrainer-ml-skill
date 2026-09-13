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

Skill id / invoke name: **`psytrainer-ml`** (PsyClaw: `/skill:psytrainer-ml`).

Turn the user's natural-language request into a PsyTrainer training or prediction run. Use only the two bundled programs for execution:

- `scripts/PsyTrainer.py` trains one or more target columns.
- `scripts/batch_predict.py` applies saved model directories to a feature CSV.

Prefer explicit `/skill:psytrainer-ml` (or an equivalent named invoke). Do not treat every mention of “train a model” as automatic activation when another analysis path is already in use.

## Install-time runtime (do this when installing the Skill, not mid-task)

Dependencies are installed once by the Skill installer (macOS / Linux / Windows).

```bash
# macOS / Linux — from the skill root after clone/copy:
python3 scripts/install_runtime.py
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
# or CMD:
scripts\install_runtime.cmd
```

The repository includes `vendor/PsyTrainer-0.2.0-cp314-none-any.whl`.
Install CPython 3.14 first. The installer selects a matching interpreter and
validates the bundled wheel checksum. Use `--recreate` to rebuild an older `.venv`.
An optional `--wheel` override supports globs expanded inside Python.

This installs the wheel and all Python dependencies, runs `pip check`, and
imports and constructs the real Trainer before writing `runtime.json` with
`ready: true`. Never use `--allow-missing-psytrainer`, report a partial install
as complete, or use a data-only dry-run as proof of readiness. A missing wheel
means an incomplete download: re-download the entire package. Offline bundles
built by `scripts/build_bundle.py` include `wheelhouse/`, detected automatically.

**At task time: do not pip install.** Require `runtime.json` to have `ready: true`,
then use its `python`. If the real Trainer cannot import, re-run the complete
installer during setup before attempting training.

## Workflow

1. Determine whether the user wants training, prediction, or interpretation of existing outputs.
2. Inspect the CSV headers and sample IDs. Ask only for choices that cannot be inferred, such as classification versus regression, target columns, or the intended scoring metric.
3. Create or update `config/ml.ini` using `config/ml.ini.example`. Paths may be absolute or relative to the config file. For a smoke check, copy `fixtures/*.csv` paths from the example comments.
4. Require `runtime.json` → `ready: true`, then use its `python`. Confirm `ccpl_training_models.trainer.Trainer` imports; if not, stop and point to `scripts/install_runtime.py`.
5. Run `--dry-run` first to validate paths, table shape, sample alignment, target/model selection, and output location.
6. Run the requested operation.
7. Read `training-summary.json` or `prediction-summary.json` and the framework's result files before explaining outcomes.

## Training

```bash
# Prefer the interpreter recorded in runtime.json
"$PSYTRAINER_PYTHON" scripts/PsyTrainer.py --config config/ml.ini --dry-run
"$PSYTRAINER_PYTHON" scripts/PsyTrainer.py --config config/ml.ini
```

Add `--target NAME` one or more times to limit training to selected label columns. The script aligns labels to features by the first CSV column, rejects duplicate or mismatched sample IDs, and rejects non-numeric or non-finite features. It creates one output directory per target and writes `training-summary.json`.

Known configuration keywords from the supplied program are `all_ff` for all feature filters, `regression` or `classification` for model families, and `all_resample` for all resampling methods. Do not invent other framework-specific tags; use values documented by the user's PsyTrainer package.

## Prediction

```bash
"$PSYTRAINER_PYTHON" scripts/batch_predict.py --config config/ml.ini --dry-run
"$PSYTRAINER_PYTHON" scripts/batch_predict.py --config config/ml.ini
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
