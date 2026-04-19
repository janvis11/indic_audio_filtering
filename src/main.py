from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io_utils import (
    ensure_dir,
    load_done_ids,
    read_manifest,
    write_csv,
    write_jsonl,
    write_parquet,
)
from .pipeline import FilteringPipeline
from .visualize import create_summary_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="JSONL manifest file OR directory of JSONL files")
    ap.add_argument("--manifest_dir", default=None, help="Alias for --manifest when passing a directory")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--disable_asr", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip sample_ids already present in metrics.parquet")
    args = ap.parse_args()

    manifest_path = args.manifest or args.manifest_dir
    if not manifest_path:
        raise ValueError("You must provide either --manifest or --manifest_dir")

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    ensure_dir(out)

    rows = read_manifest(manifest_path)

    if args.resume:
        done_ids = load_done_ids(out / "metrics.parquet")
        if done_ids:
            rows = [r for r in rows if (r.get("sample_id") or Path(r.get("audio_filepath", "")).stem) not in done_ids]

    if args.limit:
        rows = rows[:args.limit]

    pipeline = FilteringPipeline(cfg, enable_asr=not args.disable_asr)
    results = pipeline.run(rows)

    write_parquet(results, out / "metrics.parquet")
    write_csv(results, out / "metrics.csv")
    write_jsonl(results, out / "decisions.jsonl")

    for label in ["keep", "review", "reject"]:
        write_jsonl([r for r in results if r.get("decision") == label], out / f"{label}_manifest.jsonl")

    summary = {
        "total_samples": len(results),
        "keep": sum(r["decision"] == "keep" for r in results),
        "review": sum(r["decision"] == "review" for r in results),
        "reject": sum(r["decision"] == "reject" for r in results),
    }
    summary["keep_rate_pct"] = round(100.0 * summary["keep"] / max(1, summary["total_samples"]), 2)

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    create_summary_plots(str(out / "metrics.parquet"), str(out / "plots"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()