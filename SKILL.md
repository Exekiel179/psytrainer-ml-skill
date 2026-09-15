---
name: psytrainer-ml
description: Train and compare tabular CSV classification/regression models, predict, and produce diagnostic figures and Word reports. 表格建模、分类回归、批量预测、诊断图表及 Word 分析报告。
---

# PsyTrainer ML

Run from the Skill root with absolute data paths and a fresh output directory.
Use `runtime.json` -> `python` as `$PY`; if absent, follow [setup](references/setup.md).
Task CLIs check dependencies offline and install only missing/incompatible ones.
On download failure, relay the exact repair command for the user or their agent;
see setup for mirrors/offline repair. Never skip dependencies or claim success.

## Commands

- Train: `"$PY" scripts/pipeline_train.py train --features X.csv --labels Y.csv --target score --task regression --output-dir outputs/run-01`.
- Predict: `"$PY" scripts/pipeline_train.py predict --features new.csv --model outputs/run-01/pipeline.joblib --output outputs/predictions.csv`.
- Regenerate figures/report without retraining: `"$PY" scripts/pipeline_train.py report outputs/run-01`.
- Inspect uncertain inputs: `"$PY" scripts/ml.py inspect --features X.csv --labels Y.csv --task regression`.

CSV first column: unique sample ID; feature/label IDs must match. Predictors are
numeric; targets cannot be missing. Ask only about ambiguous task/target.
Defaults: 20% holdout, 5-fold CV, two models, Chinese explanatory Word report.
Use `--task classification` for classes; honor chosen models/metrics. Add a known
research question with `--question`.
For repeated subjects/sites use `--split group --metadata meta.csv --group-column subject`;
for ordered observations use `--split time --metadata meta.csv --time-column time`.

## Evidence and Context

Use raw predictors; fold-local preprocessing cannot undo upstream leakage.
Do not silently impute/recode or select models/features/thresholds from test
results. Test-driven changes need new independent test data. Load only trusted
joblib/pickle files. Diagnostics do not establish causality or significance.

Start with CLI JSON; read `analysis.json` findings only for interpretation and
log tails on failure. Report actual sample counts, metric, validation and failures.
Link generated reports/figures; do not read full CSVs, scripts or reports by
default. Scripts already write detailed beginner explanations; no LLM rewrite
is needed unless requested. Load only the relevant reference section:

- [Pipeline](references/pipeline.md): external tests, preprocessing, bounded search,
  resampling, model presets, resume and statistical limits. `ml.py capabilities`
  lists available algorithms/options.
- [Migration](references/migration.md): historical INI/parameter mapping and
  original capability audit; legacy-only [commands](references/commands.md) and
  [configuration](references/configuration.md).
