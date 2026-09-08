# psytrainer-ml

Standalone PsyClaw / Codex Skill for tabular training and batch prediction with **PsyTrainer**.

This repository is independent of the PsyClaw product tree. Install by cloning into a host Skill directory (for example `~/.psyclaw/skills/psytrainer-ml` or a project `.psyclaw/skills/`), then enable it in PsyClaw.

## Contents

| Path | Role |
|------|------|
| `SKILL.md` | Agent workflow |
| `scripts/PsyTrainer.py` | Training CLI wrapper |
| `scripts/batch_predict.py` | Batch prediction CLI |
| `config/ml.ini.example` | Config template |
| `fixtures/` | Tiny CSVs for dry-run smoke checks |
| `tests/` | Wrapper unit tests (no PsyTrainer wheel required) |

## Requirements

- A Python interpreter that can import PsyTrainer's `ccpl_training_models` (commonly 3.13 + the vendor wheel).
- `pandas`, `numpy`, `joblib` (see `requirements.txt`).
- The PsyTrainer wheel is **not** shipped here; see `NOTICE.md`.

## Quick start

```bash
cp config/ml.ini.example config/ml.ini
# edit paths, or point at fixtures/ for a dry-run shape check
python3 scripts/PsyTrainer.py --config config/ml.ini --dry-run
python3 -m unittest discover -s tests -v
```

## Security

Prediction loads `model.pkl` via joblib/pickle. Only use model directories you trust.
