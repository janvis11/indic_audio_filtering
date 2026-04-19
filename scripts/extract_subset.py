#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import json
import os
import polars as pl


def main():
    ap = argparse.ArgumentParser(description="Extract a multilingual subset from downloaded IndicVoices parquet shards")
    ap.add_argument("--hf-root", required=True, help="Root containing language parquet dirs, e.g. data/hf")
    ap.add_argument("--out-audio", required=True, help="Output folder for extracted subset audio")
    ap.add_argument("--out-manifest", required=True, help="JSONL manifest path to write")
    ap.add_argument("--langs", nargs="+", required=True, help="Languages to extract")
    ap.add_argument("--samples-per-lang", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_audio, exist_ok=True)
    records = []

    for lang in args.langs:
        lang_dir = os.path.join(args.hf_root, lang)
        parquet_files = sorted(glob.glob(os.path.join(lang_dir, "*.parquet")))
        if not parquet_files:
            print(f"Skipping {lang}: no parquet files found")
            continue

        pq = parquet_files[0]
        print(f"Reading {lang}: {pq}")

        df = pl.read_parquet(pq)
        rows = df.head(args.samples_per_lang).iter_rows(named=True)

        lang_out = os.path.join(args.out_audio, lang)
        os.makedirs(lang_out, exist_ok=True)

        saved = 0
        for i, row in enumerate(rows):
            try:
                audio_obj = row["audio_filepath"]
                audio_bytes = audio_obj["bytes"]
                audio_name = audio_obj["path"]
                ext = os.path.splitext(audio_name)[1] or ".wav"
                out_path = os.path.join(lang_out, f"{lang}_{i:04d}{ext}")
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)

                row.pop("audio_filepath", None)
                records.append({
                    "sample_id": f"{lang}_{i:04d}",
                    "audio_filepath": out_path,
                    "language": row.get("lang", lang),
                    "text": row.get("text", "")
                })
                saved += 1
            except Exception as e:
                print(f"Error in {lang} row {i}: {e}")

        print(f"Saved {saved} samples for {lang}")

    with open(args.out_manifest, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone.\nTotal samples: {len(records)}\nManifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
