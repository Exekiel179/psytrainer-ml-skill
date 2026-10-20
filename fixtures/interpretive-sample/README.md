# Interpretive report sample inputs

Synthetic regression CSV for checking report wording end to end. `score` is
generated from `signal` and `baseline`; `noise` and `age` carry no information,
and `noise` is deliberately missing more often in some rows. `context.json`
supplies labels, a unit and a tolerance.

```bash
"$PY" scripts/pipeline_train.py train \
  --features fixtures/interpretive-sample/features.csv \
  --labels fixtures/interpretive-sample/labels.csv \
  --target score --task regression --split random \
  --output-dir /tmp/interpretive-out --cv 3 --repeats 3 \
  --impute median --scaler standard --search none --metric mae \
  --language zh --question "信号与基线测量能否预测分数？" \
  --context fixtures/interpretive-sample/context.json \
  --model ModelLRRegressor --model ModelExtraTreeRegressor \
  --model ModelRandomForestRegressor --model ModelKNeighborsRegressor \
  --bootstrap 100
```
