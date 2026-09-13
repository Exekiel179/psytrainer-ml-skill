"""CSV validation shared by data inspection, training and prediction."""

import csv
from pathlib import Path


def read_table(path):
    import pandas as pd
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) < 2 or any(not name.strip() for name in header[1:]):
        raise ValueError("CSV requires an ID column and named data columns")
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV column names")
    # Preserve IDs such as 001 before pandas infers numeric column types.
    frame = pd.read_csv(path, dtype={0: str}, encoding="utf-8-sig")
    ids = frame.iloc[:, 0]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("sample IDs must be nonempty and unique")
    return frame.set_index(frame.columns[0])
