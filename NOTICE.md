# NOTICE

## This Skill

`psytrainer-ml` (this repository) is licensed under the MIT License. See `LICENSE`.

## PsyTrainer runtime

`vendor/PsyTrainer-0.2.0-cp314-none-any.whl` is the original, unmodified
PsyTrainer 0.2.0 distribution by CCPL. Its embedded
`PsyTrainer-0.2.0.dist-info/LICENSE` contains the Apache License, Version 2.0.
The wheel retains all original source files, attribution, metadata, and license.
Its SHA-256 is
`3e999045342b82cf25cb453e61a185530b553e4b9d439d2c5f5964fc5e55c357`.

The wheel declares `cp314-none-any`: the optional `--legacy` installer uses
CPython 3.14 and verifies the bundled file's checksum. The independent Pipeline
does not install or import this wheel and supports CPython 3.12-3.14.
Dependencies retain their own distribution licenses.

`scripts/model_registry.py` preserves the public model-name/estimator mapping
checked against CCPL's distribution. Its implementation calls the algorithm
libraries directly; no vendor training, scoring or parameter-grid source is
copied. The original wheel, source attribution and license remain intact.

- Prediction loads `model.pkl` via joblib/pickle: only use model directories
  produced by a trusted local training run or otherwise explicitly trusted by
  the researcher.
