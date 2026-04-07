#!/usr/bin/env python3
"""
Create a small Overleaf-ready zip so you can compile/view the LaTeX reports
without installing TeX locally.

Usage:
  python make_overleaf_bundle.py
  python make_overleaf_bundle.py --out overleaf_bundle.zip

Then:
  1) Upload the zip to Overleaf (New Project -> Upload Project)
  2) Set the main file to `final_report.tex` or `graphs_report.tex`
  3) Recompile
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="overleaf_bundle.zip")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_path = (root / args.out).resolve()

    # Keep this bundle minimal and self-contained for Overleaf.
    files = [
        "final_report.tex",
        "graphs_report.tex",
        "ann_qual_data.tex",
    ]

    missing = [p for p in files if not (root / p).exists()]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            "Missing required files for the Overleaf bundle:\n"
            f"{missing_str}\n\n"
            "If `ann_qual_data.tex` is missing, generate it with:\n"
            "  python make_ann_qual_data.py --data ann_tensors_full.pt\n"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(root / rel, arcname=rel)

    print("Wrote:", out_path)
    print("Overleaf:")
    print("  - Upload this zip as a new project")
    print("  - Set main file to `final_report.tex` or `graphs_report.tex`")


if __name__ == "__main__":
    main()
