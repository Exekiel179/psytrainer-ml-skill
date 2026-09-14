# psytrainer-ml

Standalone PsyClaw / Codex Skill for tabular training and batch prediction with **PsyTrainer**.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `/skill:psytrainer-ml`

## Download the Skill

Use `psytrainer-ml-skill.zip` from the latest GitHub Release, or clone this
repository. This is the normal Skill package: instructions, scripts, configuration,
and the bundled PsyTrainer core wheel. Run the installer once to download its
Python dependencies. At task time, use the installed environment.

The larger Windows/macOS offline ZIPs are optional alternatives for computers
without package-index access. They contain the same Skill plus cached Python
dependencies; they are not separate products and are not needed for normal setup.

This repository is independent of the PsyClaw product tree. Clone into a host Skill directory (for example project `.psyclaw/imports/recommended/psytrainer-ml` or `~/.psyclaw/skills/psytrainer-ml`), **install the runtime once**, then enable the Skill.

## Install (complete runtime required)

The original PsyTrainer 0.2.0 wheel is included in `vendor/`, with its original
license. No separate vendor download is needed. Install **CPython 3.14** first
(python.org or `uv python install 3.14`). Download the entire repository.

### macOS / Linux

```bash
git clone https://github.com/Exekiel179/psytrainer-ml-skill psytrainer-ml
cd psytrainer-ml
python3 scripts/install_runtime.py
```

### Windows

```powershell
git clone https://github.com/Exekiel179/psytrainer-ml-skill psytrainer-ml
cd psytrainer-ml
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
```

CMD equivalent:

```bat
scripts\install_runtime.cmd
```

The installer selects Python matching the wheel, creates `.venv`, and installs
the framework and **all transitive Python dependencies** in one pip resolution.
It runs `pip check`, imports the real training entry point, and constructs a
Trainer before writing `runtime.json` with `ready: true`. Any failure leaves no
success marker. `--allow-missing-psytrainer` is rejected. A data-only `--dry-run`
does not prove the runtime works.

For an older Python 3.12/3.13 environment, use `--recreate` (PowerShell:
`-Recreate`). An optional `--wheel` / `PSYTRAINER_WHEEL` override still supports
quoted globs; `--python` selects an explicit matching interpreter.

Online installation downloads third-party dependencies. On macOS, LightGBM
may require OpenMP (`brew install libomp`). Native-library import failures stop
installation and show the original error.

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

Do **not** nest this repo inside the PsyClaw source tree.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | **Install-time** venv + deps + wheel (cross-platform) |
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

For agent-driven setup and runs, use the compact CLI. Configuration generation
validates CSVs, selects a small baseline, and estimates fit count:

```bash
.venv/bin/python scripts/ml.py configure --features data/features.csv --labels data/labels.csv --task regression --target score --config config/run.ini --output-dir outputs/run-01
.venv/bin/python scripts/ml.py train --config config/run.ini
```

Use `--preset standard` for three comparison models, or repeat `--model` for
explicit choices. `ml.py capabilities` lists the installed registry on demand.
`ml.py report` returns top models without loading pickles. Detailed logs, results
and input hashes remain on disk. See [command reference](references/commands.md)
for limits and prediction setup. Existing CLI commands remain compatible.

```bash
cp config/ml.ini.example config/ml.ini
# edit paths, or point at fixtures/ for a dry-run shape check
```

```bash
# macOS / Linux
"$(python3 -c "import json; print(json.load(open('runtime.json'))['python'])")" \
  scripts/PsyTrainer.py --config config/ml.ini --dry-run
```

```powershell
# Windows PowerShell
$py = (Get-Content runtime.json | ConvertFrom-Json).python
& $py scripts\PsyTrainer.py --config config\ml.ini --dry-run
```

## Security

Prediction loads `model.pkl` via joblib/pickle. Only use model directories you trust. See `NOTICE.md` for wheel licensing.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_runtime.py
.venv/bin/python tests/smoke_agent.py
.venv/bin/python tests/smoke_pipeline.py
```

The second command runs real training and prediction on synthetic data in a
temporary directory. It requires a fully installed runtime.
