# psytrainer-ml

[English](README.md) | [简体中文](README.zh-CN.md)

Standalone PsyClaw / Claude Code / Codex Skill for tabular training, batch prediction and analysis reports.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `$psytrainer-ml` in Codex; `/skill:psytrainer-ml` in PsyClaw

## What gets installed?

This Skill combines agent instructions with local Python tools for training,
prediction, scientific figures, and Word reports. The Pipeline supports
9 classification and 12 regression algorithms, with integrated validation,
preprocessing and reporting.
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

Setup has two parts: put the **whole Skill folder** where your host discovers
skills, then run its runtime installer once. Copying `SKILL.md` alone or enabling
the Skill does not install Python dependencies. The installer creates a local
`.venv`; it does not register the Skill with your host or install Python itself.

## Let your agent install it

Give your local coding agent this request (it needs terminal and network access):

```text
Install the complete psytrainer-ml Skill from
https://github.com/Exekiel179/psytrainer-ml-skill into this host's skill directory.
Keep scripts, references, requirements.txt and the other project files together with SKILL.md.
Follow README.md to check CPython 3.12, 3.13 or 3.14 and run scripts/install_runtime.py
(or the Windows wrapper). Do not stop after downloading the instructions.
Use the Python recorded in runtime.json to run tests/smoke_pipeline.py for real
training, prediction, figures and Word output. Report the actual test result
and any unresolved installation errors.
```

## Download the Skill

Use `psytrainer-ml-skill.zip` from the [latest GitHub Release](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest), or clone this
repository. The Skill package contains instructions, scripts and configuration.
Run the installer once to download its
Python dependencies. At task time, use the installed environment.

**Download `psytrainer-ml-skill.zip`, extract the complete folder, and run the
installer.** If you clone this repository, no separate release download is needed.

Choose the destination **before** installing the runtime:

| Host / scope | Complete Skill folder |
|---|---|
| Codex, current user | `~/.agents/skills/psytrainer-ml` |
| Codex, current project | `<project>/.agents/skills/psytrainer-ml` |
| PsyClaw, current user | `~/.psyclaw/skills/psytrainer-ml` |

Codex paths follow the [official skill documentation](https://learn.chatgpt.com/docs/build-skills).
On Windows, `~` means your user profile directory. For another host, use its
configured skill directory. The destination must contain `SKILL.md` directly,
not an extra nested archive directory. This is an independent Skill repository;
PsyClaw's product source tree is not needed.

## Install (complete runtime required)

Install **CPython 3.12, 3.13 or 3.14** first (python.org or
`uv python install 3.12`). Download the entire repository. The default runtime
supports all 21 Pipeline models.

### macOS / Linux

```bash
mkdir -p "$HOME/.agents/skills"
git clone https://github.com/Exekiel179/psytrainer-ml-skill "$HOME/.agents/skills/psytrainer-ml"
cd "$HOME/.agents/skills/psytrainer-ml"
python3 scripts/install_runtime.py
```

### Windows

```powershell
New-Item -ItemType Directory -Force "$HOME/.agents/skills" | Out-Null
git clone https://github.com/Exekiel179/psytrainer-ml-skill "$HOME/.agents/skills/psytrainer-ml"
Set-Location "$HOME/.agents/skills/psytrainer-ml"
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
```

CMD equivalent:

```bat
scripts\install_runtime.cmd
```

These examples install into Codex's user skill directory; substitute your host's
destination when needed. If you downloaded a ZIP, extract it to that destination
and run the installer there; skip `git clone`. An existing installation should
be updated in place before rerunning the installer.

The installer selects a supported Python, creates `.venv`, and installs
**all transitive Python dependencies**, including LightGBM, XGBoost and CatBoost,
in one pip resolution. It runs `pip check`, imports the required libraries, and
constructs all 21 local estimators before writing `runtime.json` with
`ready: true` and `capabilities.pipeline: true`. Any failure leaves no
success marker. A data-only `--dry-run`
does not prove the runtime works.

Existing Python 3.12/3.13 environments are supported. To change the interpreter,
use `--python PATH --recreate` (PowerShell: `-Python PATH -Recreate`).

Online installation downloads third-party dependencies. On macOS, LightGBM
may require OpenMP (`brew install libomp`). Native-library import failures stop
installation and show the original error.

### Verify and start using the Skill

From the installed Skill directory, run a real end-to-end check:

```bash
# macOS / Linux
.venv/bin/python tests/smoke_pipeline.py
```

```powershell
# Windows PowerShell
& .\.venv\Scripts\python.exe tests\smoke_pipeline.py
```

This generates synthetic data and checks random/group/time validation,
prediction, figures and Word reports in a temporary directory. Installer
`ready: true` means dependency/import checks passed; this additional test checks
the actual workflow. Codex discovers installed skills automatically; restart it
if the Skill does not appear. Invoke `$psytrainer-ml` with your data and target.

The `.venv` and `runtime.json` belong to this machine and installation path.
After moving the folder, rerun the installer with `--recreate` (PowerShell:
`-Recreate`). This rebuilds `.venv`; keep datasets and results outside `.venv`.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | Complete Pipeline environment |
| `scripts/model_registry.py` | Local 21-model mapping and parameter configuration |
| `scripts/pipeline_options.py` | Local preprocessing, resampling, search and resume |
| `scripts/install_runtime.ps1` / `.cmd` | Windows wrappers |
| `scripts/pipeline_train.py` | Training, batch prediction and report commands |
| `scripts/ml.py` | Data inspection and model capability queries |
| `fixtures/` | Tiny CSVs for dry-run smoke checks |
| `tests/` | Wrapper + installer unit tests |

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

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_pipeline.py
```

The smoke test runs real training, prediction, figures and Word reports on
synthetic data using random, grouped and temporal validation.
