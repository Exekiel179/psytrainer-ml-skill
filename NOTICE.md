# NOTICE

## This Skill

`psytrainer-ml` (this repository) is licensed under the MIT License. See `LICENSE`.

## PsyTrainer runtime

Training requires the separately distributed PsyTrainer Python package
(typically a wheel such as `PsyTrainer-*-cp313-none-any.whl` providing
`ccpl_training_models`). That package is **not** bundled here.

- Install and license terms for PsyTrainer follow its own distribution.
- Do not redistribute the wheel from this repository.
- Prediction loads `model.pkl` via joblib/pickle: only use model directories
  produced by a trusted local training run or otherwise explicitly trusted by
  the researcher.
