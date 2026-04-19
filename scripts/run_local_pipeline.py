#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run the fixed project's local pipeline")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cmd = [
        args.python,
        "-m", "src.main",
        "--manifest", args.manifest,
        "--output_dir", args.output_dir,
        "--config", args.config,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
