# psytrainer-ml

Standalone PsyClaw / Codex Skill for tabular training and batch prediction with **PsyTrainer**.

- **Skill name / id:** `psytrainer-ml`
- **Invoke:** `/skill:psytrainer-ml`

This repository is independent of the PsyClaw product tree. Clone into a host Skill directory (for example project `.psyclaw/imports/recommended/psytrainer-ml` or `~/.psyclaw/skills/psytrainer-ml`), **install the runtime once**, then enable the Skill.

## Install (dependencies happen here)

```bash
git clone https://github.com/Exekiel179/psytrainer-ml-skill psytrainer-ml
cd psytrainer-ml
python3 scripts/install_runtime.py --wheel /path/to/PsyTrainer-*-cp313-none-any.whl
```

`install_runtime.py` creates `.venv/`, installs `requirements.txt` (pandas/numpy/joblib), installs the PsyTrainer wheel, and writes `runtime.json`. Pass `PSYTRAINER_WHEEL=...` instead of `--wheel` if preferred.

Do **not** nest this repo inside the PsyClaw source tree.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/install_runtime.py` | **Install-time** venv + deps + wheel |
| `scripts/PsyTrainer.py` | Training CLI wrapper |
| `scripts/batch_predict.py` | Batch prediction CLI |
| `config/ml.ini.example` | Config template |
| `fixtures/` | Tiny CSVs for dry-run smoke checks |
| `tests/` | Wrapper unit tests (no PsyTrainer wheel required) |

## Quick start (after install)

```bash
cp config/ml.ini.example config/ml.ini
# edit paths, or point at fixtures/ for a dry-run shape check
$(python3 -c "import json; print(json.load(open('runtime.json'))['python'])") \
  scripts/PsyTrainer.py --config config/ml.ini --dry-run
```

## Security

Prediction loads `model.pkl` via joblib/pickle. Only use model directories you trust. See `NOTICE.md` for wheel licensing.
