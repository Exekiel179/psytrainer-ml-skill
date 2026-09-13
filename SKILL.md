---
name: psytrainer-ml
description: >-
  Inspect feature/label CSVs, configure PsyTrainer classification or regression,
  train leakage-aware tabular Pipelines with group/time validation, predict,
  and export measured figures and Word reports.
metadata:
  short-description: Inspect, train and predict with PsyTrainer
---

# PsyTrainer ML

Use for requested PsyTrainer work; invoke `/skill:psytrainer-ml` or `$psytrainer-ml`.
Use `runtime.json` -> `python` as `$PY`; require `ready: true`. Missing runtime:
follow the installation section of [README.md](README.md) during setup. Download
the whole Skill including `vendor/`; never bypass missing dependencies.

Prefer `scripts/pipeline_train.py train --features X.csv --labels Y.csv --target score --task regression --output-dir outputs/run-01` for new analyses. It handles fold-local preprocessing, model selection, held-out evaluation, figures and `report.docx` in one call. Defaults: 20% internal holdout, 5 folds, two base models, Chinese report. Set `--question` when the research question is known.

For subjects/sites use `--split group --metadata meta.csv --group-column subject`;
for longitudinal data use `--split time --metadata meta.csv --time-column time`.
Read [pipeline.md](references/pipeline.md) only for external tests, preprocessing,
explanation limits or output options. Prediction uses this same script's `predict`
command and `pipeline.joblib`; legacy pickles use the legacy prediction command.
Never select models/features using held-out results or importance. Require raw
predictors: a Pipeline cannot undo leakage introduced upstream. Explain missing
classes, model failures and validation limits reported in `summary.json`.

Use `scripts/ml.py` for legacy INI workflows and installed model discovery. It validates inputs and returns compact JSON. Full training
logs and results stay on disk. Do not read scripts, entire CSVs or all references
by default. Consult [commands.md](references/commands.md) for options and limits,
or [configuration.md](references/configuration.md) for custom INI settings.

- Known task and paths: `"$PY" scripts/ml.py configure --features X.csv --labels Y.csv --task regression --target score --config config/run.ini --output-dir outputs/run-01`. This includes data inspection and a two-model baseline.
- Need data diagnosis first: `"$PY" scripts/ml.py inspect --features X.csv --labels Y.csv --task regression`.
- Run: `"$PY" scripts/ml.py train --config config/run.ini`; use `predict` for a prediction INI.
- Existing results: `"$PY" scripts/ml.py report outputs/run-01/training-summary.json --top 3`.
- Need supported algorithms/metrics: `"$PY" scripts/ml.py capabilities`.

Ask for task type or target only when ambiguous. Preserve explicit model/metric
choices. Resolve reported errors before training; never silently impute or recode.
Use fresh output directories. On failure inspect the referenced log tail.

Only load trusted local model pickles/joblib files. Summarize sample count, target, primary
metric, validation design and failures using actual output; link the report and
figures instead of reading their full contents into context. Compare scores only
within a target/metric. The following limits apply to the legacy runner only:
vendor losses are negated; R2 selection is defective and
disabled in the new runner. Filtering/resampling before CV can leak information;
random folds do not support grouped or temporal validation. Do not claim held-out
performance, causal effects, confidence intervals or reproducibility not measured.
