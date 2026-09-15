# psytrainer-ml

[English](README.md) | [简体中文](README.zh-CN.md)

Standalone PsyClaw / Claude Code / Codex Skill for tabular training, batch prediction and analysis reports.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `/psytrainer-ml` in Claude Code; `$psytrainer-ml` in Codex; `/skill:psytrainer-ml` in PsyClaw

## Capabilities

This Skill combines agent instructions with local Python tools for training,
prediction, scientific figures, and Word reports. The Pipeline supports
9 classification and 12 regression algorithms, with integrated validation,
preprocessing and reporting.
Reports explain the task, data splits, preprocessing and metrics for readers
without machine-learning experience. Each diagnostic figure includes a reading
guide, available measured findings and interpretation limits.
It also provides bounded grid/random search, forward/F-threshold selection,
seven resampling methods, domain model presets and checked CV resume.

### Local preprocessing

| Operation | Local Pipeline option |
|---|---|
| Missing values | Reject by default; `--impute median` fits training-fold medians |
| Scaling | StandardScaler by default; `--scaler minmax` or `--no-scale` |
| Univariate selection | `--select-k N` or `--selection f-threshold --f-threshold 5` |
| Forward selection | `--selection forward --select-k N`, with group/time-aware inner CV |
| PCA | `--pca N`, saved with the fitted preprocessing for prediction |
| Classification resampling | `random-over`, `smote`, `smote-tomek`, `smote-enn`, `cluster-centroids`, `random-under`, `near-miss` via `--resample`; training only |
| Parameter/preprocessing search | `--search grid` or `random`, conditional `--search-space`, per-model `--max-candidates` |
| Domain model sets | `--preset audio`, `face`, `gait`, `text`: regression on extracted numeric features, not raw-media extraction |
| Resume and persistence | `--resume` verifies inputs/configuration/versions; `--save-candidates` retains each model's best configuration |

Learned transforms fit within training folds. The holdout is excluded from
selection/search; figures and Word reports record the actual processing and results.

Place the **whole Skill folder** in your host's skill directory and run
`scripts/install_runtime.py` once. It creates `.venv` without downloading model
libraries. Each task checks its dependencies and installs missing or incompatible
packages before work. Load the Skill through your host and start using it;
sample training is not part of user setup.

## Let your agent install it

Give your local coding agent this request (it needs terminal and network access):

```text
Install the latest Release Skill ZIP from https://github.com/Exekiel179/psytrainer-ml-skill following references/setup.md.
```

For Claude Code, use the explicit destination:

```text
Install the latest Release Skill ZIP from https://github.com/Exekiel179/psytrainer-ml-skill into ~/.claude/skills/psytrainer-ml and run scripts/install_runtime.py.
```

Agents can use the [short setup guide](references/setup.md); load `SKILL.md` for
tasks and read other references only as needed. Detailed Word explanations are
generated locally, so they do not require repeated model calls or report ingestion.

## Download the Skill

Use `psytrainer-ml-skill.zip` from the [latest GitHub Release](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest).
It contains runtime instructions, scripts and configuration; development tests
and sample datasets are excluded. Cloning the repository is an alternative for
developers and includes those files. Run the runtime setup script after extraction.

Choose the destination **before** installing the runtime:

| Host / scope | Complete Skill folder |
|---|---|
| Claude Code, current user | `~/.claude/skills/psytrainer-ml` |
| Claude Code, current project | `<project>/.claude/skills/psytrainer-ml` |
| Codex, current user | `~/.agents/skills/psytrainer-ml` |
| Codex, current project | `<project>/.agents/skills/psytrainer-ml` |
| PsyClaw, current user | `~/.psyclaw/skills/psytrainer-ml` |

Codex paths follow the [official skill documentation](https://learn.chatgpt.com/docs/build-skills).
On Windows, `~` means your user profile directory. For another host, use its
configured skill directory. The destination must contain `SKILL.md` directly,
not an extra nested archive directory. This is an independent Skill repository;
PsyClaw's product source tree is not needed.

Claude Code reads `.claude/skills`, not Codex's `.agents/skills`. Downloading this
repository into an ordinary project folder or only configuring Python does not
register a Claude Code skill. See [Claude Code skill locations](https://code.claude.com/docs/en/skills#choose-where-skills-load).

## Setup

Install **CPython 3.12, 3.13 or 3.14** first (python.org or
`uv python install 3.12`). Extract the complete `psytrainer-ml` folder from the
Release ZIP into the chosen host directory. Git is not required. The default runtime
supports all 21 Pipeline models.

The following commands use Claude Code's user directory. For Codex substitute
`$HOME/.agents/skills/psytrainer-ml`; for PsyClaw use `$HOME/.psyclaw/skills/psytrainer-ml`.

### macOS / Linux

```bash
cd "$HOME/.claude/skills/psytrainer-ml"
python3 scripts/install_runtime.py
```

### Windows

```powershell
Set-Location "$HOME/.claude/skills/psytrainer-ml"
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
```

CMD equivalent:

```bat
scripts\install_runtime.cmd
```

These commands configure the runtime after extraction to Claude Code's user skill directory;
substitute your host's destination when needed. To upgrade, replace the Skill files,
preserve `.venv` and your data/results, and rerun setup.

The setup script selects a supported Python and creates `.venv`. `runtime.json`
with `ready: true` records that the environment exists, not that every package is
installed. Every task invocation checks current dependency versions and imports.
Satisfied dependencies require no network downloads.

Default training installs core modeling, plotting and Word report libraries.
LightGBM, XGBoost and CatBoost are installed only when selected; resampling adds
imbalanced-learn. Prediction repairs dependencies required by the saved model.
Report regeneration does not install unrelated model libraries. Use
`--packages xgboost` to preinstall a specific package, or `--all` only for explicit
full preinstallation and release QA. Setup never runs sample training.

Existing Python 3.12/3.13 environments are supported. To change the interpreter,
use `--python PATH --recreate` (PowerShell: `-Python PATH -Recreate`).

Online installation downloads third-party dependencies. On macOS, LightGBM
may require OpenMP (`brew install libomp`). Native-library import failures stop
installation and show the original error.

### Load the Skill

Invoke `/psytrainer-ml` in Claude Code, `$psytrainer-ml` in Codex, or
`/skill:psytrainer-ml` in PsyClaw. Claude Code can also select the Skill for requests
about tabular model training, prediction and Word analysis reports.

If Claude Code does not recognize it, confirm that
`~/.claude/skills/psytrainer-ml/SKILL.md` exists, then open a new session and type
`/psytrainer-ml`. Project installations only apply within their project scope.
If the path is correct, check skill-disable settings or a conflicting skill name.
Reinstalling Python dependencies does not repair host discovery.

The `.venv` and `runtime.json` belong to this machine and installation path.
After moving the folder, rerun the runtime setup script with `--recreate` (PowerShell:
`-Recreate`). This rebuilds `.venv`; keep datasets and results outside `.venv`.

### Slow or interrupted networks

Choose a reachable package index for this run, for example the
[Tsinghua PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) in China:

```bash
python3 scripts/install_runtime.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1 -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
```

- pip is the default and retains its cache and index settings. Explicit index, installer and timeout options are saved in this Skill's `runtime.json` for subsequent task repairs, without changing global settings.
- With uv already installed, add `--installer uv` (PowerShell: `-Installer uv`) for concurrent downloads and caching. uv uses separate configuration/cache and does not read pip settings; pass `--index-url` when needed. See [uv documentation](https://docs.astral.sh/uv/pip/compatibility/).
- Defaults: 20-second network timeout, 2 retries, 600-second installation budget. Adjust `--timeout`, `--retries`, `--max-seconds` (PowerShell: `-Timeout`, `-Retries`, `-MaxSeconds`).
- After interruption, rerun the same command to reuse installed packages and completed cached downloads. Incomplete files may need downloading again. Do not use `--recreate` for routine retries. See [pip caching](https://pip.pypa.io/en/stable/topics/caching/).
- Check selected packages offline: `python3 scripts/install_runtime.py --check --packages xgboost`. Plain `--check` checks the environment; `--check --all` checks every dependency. PowerShell: `-Check -Packages xgboost` or `-Check -All`.
- With a dependency directory matching the target platform and Python, use `--wheelhouse PATH` (PowerShell: `-Wheelhouse PATH`) for offline installation.

Only prebuilt packages are accepted to avoid lengthy source builds. A missing
compatible package fails explicitly. Maintainers test real training, prediction
and reports before release.

When dependency installation fails, the task stops and returns the required
packages and a repair command targeting the correct interpreter. The agent must
show that command for manual installation or execute it for the user, select a
reachable `--index-url` or local `--wheelhouse`, and rerun the original task after
repair. Reporting a network failure alone is insufficient; never skip required
packages or report task success. For example:

```bash
python3 scripts/install_runtime.py --packages xgboost --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | Python runtime setup and dependency checks |
| `scripts/model_registry.py` | Local 21-model mapping and parameter configuration |
| `scripts/pipeline_options.py` | Local preprocessing, resampling, search and resume |
| `scripts/install_runtime.ps1` / `.cmd` | Windows wrappers |
| `scripts/pipeline_train.py` | Training, batch prediction and report commands |
| `scripts/ml.py` | Data inspection and model capability queries |

## Quick start (after install)

Run these commands from the installed Skill directory. Paths such as `data/`
are placeholders for your own files; absolute paths work too. On Windows replace
`.venv/bin/python` with `& .\.venv\Scripts\python.exe` in PowerShell.

Use the fold-local Pipeline runner, which generates plots and a
Word report automatically:

```bash
.venv/bin/python scripts/pipeline_train.py train --features data/features.csv --labels data/labels.csv --task regression --target score --output-dir outputs/run-01
.venv/bin/python scripts/pipeline_train.py predict --features data/new.csv --model outputs/run-01/pipeline.joblib --output outputs/predictions.csv
```

Supports grouped/time validation, an independent holdout or external test set,
fold-local imputation/selection/PCA/resampling, permutation importance, vector
figures and Chinese/English `.docx` reports. See [Pipeline reference](references/pipeline.md).

Reports include training/validation gaps, a development-fitted dummy baseline,
paired test gains and supported bootstrap intervals, regression error structure,
classification calibration and threshold trade-offs, input shift, correlated
feature reliance, and group/time stability. Each finding links to numeric source
data; analysis and report generation use no LLM calls. These diagnostics guide
development-set experiments and do not automatically tune against test results.

Use `--search grid` or `--search random --max-candidates 12` for tuning;
`--search-space FILE.json` defines model and preprocessing combinations.
`--selection forward --select-k 3`, `--selection f-threshold`, `--scaler minmax`,
the seven `--resample` methods, `--preset all|audio|face|gait|text`, `--resume`
and `--save-candidates` are documented in the [training options](references/pipeline.md#migrated-training-options).

Use `scripts/ml.py capabilities` to list supported models and processing options.

## Security

Prediction loads `pipeline.joblib` via joblib/pickle. Only load models from trusted sources.
This Skill uses the [MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for third-party notices.

## Development Verification

Run these commands in the source checkout. Release downloads omit `tests/` and `fixtures/`.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_pipeline.py
```

The smoke test runs real training, prediction, figures and Word reports on
synthetic data using random, grouped and temporal validation.

For maintainers, the [release procedure](references/releasing.md) describes
automated archive construction, cross-platform verification and publication.
