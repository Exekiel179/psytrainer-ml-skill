# NOTICE

## This Skill

`psytrainer-ml` (this repository) is licensed under the MIT License. See `LICENSE`.

## Historical compatibility reference

The local model mappings and migration audit reference PsyTrainer 0.2.0 by CCPL.
The audited distribution was `PsyTrainer-0.2.0-cp314-none-any.whl`, whose embedded
license was Apache License, Version 2.0. Its SHA-256 was
`3e999045342b82cf25cb453e61a185530b553e4b9d439d2c5f5964fc5e55c357`.

This version does not distribute, download or require the PsyTrainer wheel.
The independent Pipeline supports CPython 3.12-3.14. Historical INI compatibility
requires a separately supplied external wheel and its declared Python version.
Dependencies retain their own distribution licenses.

`scripts/model_registry.py` preserves the public model-name/estimator mapping
checked against CCPL's distribution. Its implementation calls the algorithm
libraries directly; no vendor training, scoring or parameter-grid source is
copied. Original source attribution in retained compatibility wrappers is preserved.

- Prediction loads `model.pkl` via joblib/pickle: only use model directories
  produced by a trusted local training run or otherwise explicitly trusted by
  the researcher.
