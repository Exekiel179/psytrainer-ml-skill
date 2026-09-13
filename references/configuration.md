# Configuration notes

## `config/ml.ini`

Copy from `config/ml.ini.example`. Paths are resolved relative to the config file when not absolute.

### `[PsyTrainer]`

| Key | Notes |
|-----|--------|
| `feature_file` / `label_file` | CSV with sample ID as the first/index column |
| `output_dir` | One subdirectory per target |
| `cv_times` | Must be ≥ 2 |
| `ff_tags` / `model_tags` / `resample_tags` | Comma-separated; use PsyTrainer-documented tags only |
| `scoring` | Optional comma-separated metric names |
| `targets` | Optional comma-separated labels; CLI `--target` overrides |

For new configurations, prefer `ml.py configure`; it validates data and emits
small model presets. See [commands.md](commands.md) for advanced settings and
vendor validation limitations. Legacy `all_ff` requires a positive `pca_num` and
may yield optimistic CV scores; it is not the default in generated configurations.

### `[batch_predict]`

| Key | Notes |
|-----|--------|
| `feature_file` | Rows preserved; missing/non-finite values rejected |
| `model_root` | Immediate children are model directories; unfiltered `NoneType&...` estimators may provide saved feature names instead of `feature_filter.csv` |
| `output_dir` | One prediction CSV per successful model |

## Trust

Do not point `model_root` at untrusted downloads. Pickle deserialization can execute code.
