---
name: psytrainer-ml
description: Train and compare tabular CSV classification/regression models, predict, and produce diagnostic figures and Word reports. 表格建模、分类回归、批量预测、诊断图表及 Word 分析报告。
---

# PsyTrainer ML

## Setup

Run from the Skill root with absolute data paths and a fresh output directory.
Use `runtime.json` -> `python` as `$PY`; if absent, follow [setup](references/setup.md).
Task CLIs check dependencies offline and install only missing/incompatible ones.
On download failure, relay the exact repair command for the user or their agent;
see setup for mirrors/offline repair. Never skip dependencies or claim success.

This Skill ships a **Pipeline-only** path (`scripts/pipeline_train.py`). There is
no optional PsyTrainer wheel / `vendor/` / legacy INI runtime.

## Commands

- Train: `"$PY" scripts/pipeline_train.py train --features X.csv --labels Y.csv --target score --task regression --output-dir outputs/run-01`.
- Predict: `"$PY" scripts/pipeline_train.py predict --features new.csv --model outputs/run-01/pipeline.joblib --output outputs/predictions.csv`.
- Regenerate figures/report without retraining: `"$PY" scripts/pipeline_train.py report outputs/run-01 [--context ctx.json] [--language en] [--analysis full]`.
- Inspect uncertain inputs: `"$PY" scripts/ml.py inspect --features X.csv --labels Y.csv --task regression`.

CSV first column: unique sample ID; feature/label IDs must match. Predictors are
numeric; targets cannot be missing. Ask only about ambiguous task/target.
Defaults: 20% holdout, 5-fold CV, two models, Chinese technical Word report at the
**basic** level. The report is findings-first: a structured abstract (objective /
methods / results / conclusion) and numbered key findings answer the research
question, then methods, results (model comparison, test performance, error
structure, feature importance, strata), a short discussion, limitations, next
steps and appendices (per-model fold results, metric definitions, figure-reading
guide, reproducibility). Basic reports state test metrics as plain values and
omit the dummy-baseline comparison, bootstrap 95% CIs and training/test
comparability checks. Use `--analysis full` (train or report) only when the user
asks for these; `report DIR --analysis full` adds them without retraining.
With ≥4 successful models, per-model details go to `reports/<Model>.md` (+ `.docx`).
For binary outcomes, set `--positive-class` to the clinically meaningful class
(e.g. `case`, `depressed`) — otherwise the alphabetically second class is
"positive" for AUC/calibration, which can be the control group. Ask if unclear.
When the user wants a significance statement ("better than chance", p-value),
add `--permutations 1000` and tell them it refits the final model 1000 times.
Use `--task classification` for classes; honor chosen models/metrics. Add a known
research question with `--question`.

## Report conventions

Reports follow the reporting style of machine-learning studies in psychology and
behavioural science, and any summary or rewrite you produce must keep it:

- Terms: 训练集 / 测试集 (training / test set), 交叉验证 (CV), 最终模型 (final
  model), 置换重要性 (permutation importance). Never "开发集", "配对改善" or
  "正向重要性".
- Statistics in APA form with two decimals: `MAE = 0.45, R² = 0.99`; CV results
  as `M = 0.43, SD = 0.10`; importance as `ΔMAE = 4.84, SD = 0.25`.
- p-values only come from the optional permutation test, APA style (`p = .005`,
  `p < .001`).
- Full level only: 基线模型 (`DummyRegressor` predicting the training median or
  mean; `DummyClassifier` the majority class), test 95% CI = bootstrap percentile
  interval (`MAE = 0.16, 95% CI [0.11, 0.21]`), improvement "MAE 降低了 90%（差值 =
  1.55, 95% CI [0.98, 2.09]）"; the difference CI excluding 0 means "outperforms
  the baseline".
- Tree ensembles (random forest, extra trees) fit training folds almost perfectly
  by design; do not call that overfitting.
- Regression conclusions state variance explained from R² ("可解释约 99% 的变异").
- Figures are always in English (for international journals), with model display
  names and real class names; report text follows `--language`.

## Report context

Reports read best when they know what the variables mean. When the user has said
(or readily knows) the outcome's meaning or unit, predictor meanings, class
meanings, or an acceptable prediction error, write a JSON file and pass
`--context ctx.json` to `train` or `report`; it is saved as `report-context.json`
and reused on later regenerations. Keys (all optional): `question`, `title`,
`background`, `population`, `target_label`, `target_unit`, `tolerance` (positive,
target units; reports the share of test records within it), `feature_labels`
`{column: label}`, `class_labels` `{class: label}`. Only record what the user
stated; never invent units, tolerances or meanings. If none is known, train
without it and offer once to regenerate with context afterwards.

For repeated subjects/sites use `--split group --metadata meta.csv --group-column subject`;
for ordered observations use `--split time --metadata meta.csv --time-column time`.

## Evidence and Context

Use raw predictors; fold-local preprocessing cannot undo upstream leakage.
Do not silently impute/recode or select models/features/thresholds from test
results. Test-driven changes need new independent test data. Load only trusted
joblib/pickle files. Diagnostics do not establish causality or significance.

Start with CLI JSON. To tell the user the outcome, read only the `## 摘要` /
`## Abstract` and `## 要点` / `## Key findings` sections at the top of `report.md`
(they hold the verdict, key numbers and triggered issues) and relay them in a few
sentences: whether the features predicted the outcome, the primary test metric
(with its interval and baseline only at the full level), the leading predictors,
and any flagged issue. Read log tails only on failure. Link the generated
reports/figures; do not read full CSVs, scripts or whole reports by default. The
scripts already write the full report; no LLM rewrite is needed unless requested,
and any rewrite must keep the measured numbers and stated limits. Load only the
relevant reference section:

- [Setup](references/setup.md): Pipeline runtime (`.venv` / `runtime.json`).
- [Pipeline](references/pipeline.md): external tests, preprocessing, bounded search,
  resampling, model presets, resume, report layout and statistical limits.
  `ml.py capabilities` lists available algorithms/options.
- [Migration](references/migration.md): historical notes; Pipeline is the only path.
