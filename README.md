# psytrainer-ml

Standalone PsyClaw / Codex Skill for tabular training and batch prediction with **PsyTrainer**.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `/skill:psytrainer-ml`

This repository is independent of the PsyClaw product tree. Clone into a host Skill directory (for example project `.psyclaw/imports/recommended/psytrainer-ml` or `~/.psyclaw/skills/psytrainer-ml`), **install the runtime once**, then enable the Skill.

## Install (dependencies happen here)

### macOS / Linux

```bash
git clone https://github.com/Exekiel179/psytrainer-ml-skill psytrainer-ml
cd psytrainer-ml
python3 scripts/install_runtime.py --wheel /path/to/PsyTrainer-*-cp313-none-any.whl
```

### Windows

```powershell
git clone https://github.com/Exekiel179/psytrainer-ml-skill psytrainer-ml
cd psytrainer-ml
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1 -Wheel D:\wheels\PsyTrainer-*.whl
```

CMD equivalent:

```bat
scripts\install_runtime.cmd --wheel D:\wheels\PsyTrainer-0.2.0-cp313-none-any.whl
```

`install_runtime.py` creates `.venv\`, installs `requirements.txt` (pandas/numpy/joblib), installs the PsyTrainer wheel, and writes `runtime.json`. Wheel globs work on Windows because expansion happens in Python. Prefer Python 3.13 via the `py` launcher when the wheel is `cp313`.

Do **not** nest this repo inside the PsyClaw source tree.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | **Install-time** venv + deps + wheel (cross-platform) |
| `scripts/install_runtime.ps1` / `.cmd` | Windows wrappers |
| `scripts/PsyTrainer.py` | Training CLI wrapper |
| `scripts/batch_predict.py` | Batch prediction CLI |
| `config/ml.ini.example` | Config template |
| `fixtures/` | Tiny CSVs for dry-run smoke checks |
| `tests/` | Wrapper + installer unit tests |

## Quick start (after install)

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
