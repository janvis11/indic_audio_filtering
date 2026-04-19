from __future__ import annotations

import json
from pathlib import Path

import polars as pl

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_manifest(path):
    path = Path(path)
    if path.is_dir():
        rows = []
        for file in sorted(path.rglob("*.jsonl")):
            rows.extend(pl.read_ndjson(file).to_dicts())
        return rows
    return pl.read_ndjson(path).to_dicts()

def write_jsonl(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)

    flat_rows = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        flat_rows.append(flat)

    pl.DataFrame(flat_rows).write_csv(path)

def write_parquet(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    pl.DataFrame(rows).write_parquet(path)

def load_done_ids(metrics_path):
    """
    Read existing metrics parquet/csv and return already-processed sample_ids.
    Used for --resume.
    """
    path = Path(metrics_path)
    if not path.exists():
        return set()

    try:
        if path.suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path)
        if "sample_id" not in df.columns:
            return set()
        return set(df["sample_id"].to_list())
    except Exception:
        return set()