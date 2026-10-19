# Interpretive report sample inputs

Synthetic regression CSV used to regenerate the v0.7.1 interpretive report sample
(`media/v0.7.1-interpretive-report.md` / `.docx` in the project store).

```bash
"$PY" scripts/pipeline_train.py train \
  --features fixtures/interpretive-sample/features.csv \
  --labels fixtures/interpretive-sample/labels.csv \
  --target score --task regression --split random \
  --output-dir /tmp/interpretive-out --cv 3 --repeats 3 \
  --impute median --scaler standard --search none --metric mae \
  --language zh --question "样例：信号与基线能否预测分数？" \
  --model ModelLRRegressor --model ModelExtraTreeRegressor \
  --model ModelRandomForestRegressor --model ModelKNeighborsRegressor \
  --bootstrap 100
```
