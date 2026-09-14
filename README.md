# psytrainer-ml

[English](README.md) | [简体中文](README.zh-CN.md)

Standalone PsyClaw / Codex Skill for tabular training and batch prediction with **PsyTrainer**.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `$psytrainer-ml` in Codex; `/skill:psytrainer-ml` in PsyClaw

## What gets installed?

This Skill combines agent instructions with local Python tools for training,
prediction, scientific figures, and Word reports. The bundled `.whl` is the
optional original PsyTrainer Python library, not a second Skill or a pretrained
model. The Pipeline now uses a local registry of the same 9 classification and
12 regression algorithms, calling their libraries directly. It retains model
names and estimator defaults while owning validation, preprocessing and reporting.

Setup has two parts: put the **whole Skill folder** where your host discovers
skills, then run its runtime installer once. Copying `SKILL.md` alone or enabling
the Skill does not install Python dependencies. The installer creates a local
`.venv`; it does not register the Skill with your host or install Python itself.

## Let your agent install it

Give your local coding agent this request (it needs terminal and network access):

```text
Install the complete psytrainer-ml Skill from
https://github.com/Exekiel179/psytrainer-ml-skill into this host's skill directory.
Keep scripts, references, requirements.txt and vendor together with SKILL.md.
Follow README.md to check CPython 3.12, 3.13 or 3.14 and run scripts/install_runtime.py
(or the Windows wrapper). Do not stop after downloading the instructions.
Use the Python recorded in runtime.json to run tests/smoke_pipeline.py for real
training, prediction, figures and Word output. Report the actual test result
and any unresolved installation errors.
```

## Download the Skill

Use `psytrainer-ml-skill.zip` from the [latest GitHub Release](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest), or clone this
repository. This is the normal Skill package: instructions, scripts, configuration,
and the optional original PsyTrainer wheel. Run the installer once to download its
Python dependencies. At task time, use the installed environment.

The larger Windows/macOS offline ZIPs are optional alternatives for computers
without package-index access. They contain the same Skill plus cached Python
dependencies; they are not separate products and are not needed for normal setup.

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
supports every Pipeline model without installing PsyTrainer. Its original wheel
remains included in `vendor/` for optional INI compatibility.

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
success marker. `--allow-missing-psytrainer` is rejected. A data-only `--dry-run`
does not prove the runtime works.

Existing Python 3.12/3.13 environments are supported. To change the interpreter,
use `--python PATH --recreate` (PowerShell: `-Python PATH -Recreate`).
The optional `--legacy` mode has its own environment and version requirements,
described below. `--wheel` / `PSYTRAINER_WHEEL` imply that mode.

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

### Offline bundle

Build on a connected machine. This downloads the entire resolved Python
dependency set and creates a ZIP only after every download succeeds. Windows x64:

```bash
python3 scripts/build_bundle.py --platform win_amd64 --output dist/psytrainer-ml-windows-x64.zip
```

Omit `--platform` for this machine's platform. Bundles are specific to the target
OS, architecture, and CPython version. They do not include the Python interpreter
or OS shared libraries. The source ZIP includes the core wheel but is not an
offline dependency bundle.

Extract the entire bundle and run the normal installer. It detects `wheelhouse/`
and uses `--no-index` without network fallback. Alternatively pass
`--wheelhouse PATH` (PowerShell: `-Wheelhouse PATH`). Missing dependencies fail
installation; a partial install is never reported as ready.

Use `--python python3.12` to choose the bundle's Python version explicitly.
`bundle.json` records that version and the installer selects it on the target.
Add `--legacy` when building dependencies for the original INI engine, and run
the target installer with `--legacy` too. A Pipeline-only bundle cannot install
the original engine offline.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | Complete Pipeline environment; optional separate original engine |
| `scripts/model_registry.py` | Local 21-model mapping and parameter configuration |
| `vendor/` | Original PsyTrainer wheel and embedded license |
| `scripts/build_bundle.py` | Complete offline Python dependency ZIP builder |
| `scripts/install_runtime.ps1` / `.cmd` | Windows wrappers |
| `scripts/PsyTrainer.py` | Training CLI wrapper |
| `scripts/ml.py` | Compact agent commands: inspect, configure, capabilities, train, predict, report |
| `scripts/batch_predict.py` | Batch prediction CLI |
| `config/ml.ini.example` | Config template |
| `fixtures/` | Tiny CSVs for dry-run smoke checks |
| `tests/` | Wrapper + installer unit tests |

## Quick start (after install)

Run these commands from the installed Skill directory. Paths such as `data/`
are placeholders for your own files; absolute paths work too. On Windows replace
`.venv/bin/python` with `& .\.venv\Scripts\python.exe` in PowerShell.

New analyses use the fold-local Pipeline runner, which generates plots and a
Word report automatically:

```bash
.venv/bin/python scripts/pipeline_train.py train --features data/features.csv --labels data/labels.csv --task regression --target score --output-dir outputs/run-01
.venv/bin/python scripts/pipeline_train.py predict --features data/new.csv --model outputs/run-01/pipeline.joblib --output outputs/predictions.csv
```

Supports grouped/time validation, an independent holdout or external test set,
fold-local imputation/selection/PCA/resampling, permutation importance, vector
figures and Chinese/English `.docx` reports. See [Pipeline reference](references/pipeline.md).

Reports now include training/validation gaps, a development-fitted dummy baseline,
paired test gains and supported bootstrap intervals, regression error structure,
classification calibration and threshold trade-offs, input shift, correlated
feature reliance, and group/time stability. Each finding links to numeric source
data; analysis and report generation use no LLM calls. These diagnostics guide
development-set experiments and do not automatically tune against test results.
Existing installations should rerun the installer to add the report dependencies.

### Legacy INI workflows

Use this section for existing PsyTrainer INI jobs. New analyses should use the
Pipeline commands above; these two runners have different validation behavior.

Install the optional original engine once:

```bash
python3 scripts/install_runtime.py --legacy
```

On Windows use `scripts\install_runtime.cmd --legacy` or the PowerShell wrapper
with `-Legacy`. The bundled original wheel requires CPython 3.14 **only for this
mode**. It is checksum-verified and installed into `.venv-legacy`; the separate
`runtime-legacy.json` records its interpreter and `capabilities.legacy: true`.
The main `.venv` and `runtime.json` are preserved. Existing pre-migration
environments containing PsyTrainer still work; no package is removed automatically.

For agent-driven setup and runs, use the compact CLI. Configuration generation
validates CSVs, selects a small baseline, and estimates fit count:

```bash
.venv-legacy/bin/python scripts/ml.py configure --features data/features.csv --labels data/labels.csv --task regression --target score --config config/run.ini --output-dir outputs/run-01
.venv-legacy/bin/python scripts/ml.py train --config config/run.ini
```

Use `--preset standard` for three comparison models, or repeat `--model` for
explicit choices. `ml.py capabilities --legacy` lists the original registry;
`ml.py capabilities` lists the independent Pipeline registry.
`ml.py report` returns top models without loading pickles. Detailed logs, results
and input hashes remain on disk. See [command reference](references/commands.md)
for limits and prediction setup. Existing CLI commands remain compatible.

```bash
cp config/ml.ini.example config/ml.ini
# edit paths, or point at fixtures/ for a dry-run shape check
```

```bash
# macOS / Linux
"$(python3 -c "import json; print(json.load(open('runtime-legacy.json'))['python'])")" \
  scripts/PsyTrainer.py --config config/ml.ini --dry-run
```

```powershell
# Windows PowerShell
$py = (Get-Content runtime-legacy.json | ConvertFrom-Json).python
& $py scripts\PsyTrainer.py --config config\ml.ini --dry-run
```

## Security

Prediction loads `model.pkl` via joblib/pickle. Only use model directories you trust. See `NOTICE.md` for wheel licensing.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_pipeline.py
# Optional original engine, after --legacy installation
.venv-legacy/bin/python tests/smoke_runtime.py
.venv-legacy/bin/python tests/smoke_agent.py
```

The smoke commands run real training and prediction on synthetic data. See
[migration and compatibility](references/migration.md) for the model mapping,
parameter configuration, preserved behavior and model-file limitations.
