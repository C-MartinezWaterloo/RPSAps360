#!/usr/bin/env python3
"""
Open the interactive graphs dashboard (no LaTeX required).

Usage:
  python open_graphs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    html = root / "plots" / "report_graphs.html"
    if not html.exists():
        print("Missing:", html)
        print("Generate it with:")
        print("  python make_report_graphs.py --results results_all.csv --out plots/report_graphs.html")
        sys.exit(1)

    url = html.resolve().as_uri()
    print("Graphs HTML:", html)
    print("URL:", url)

    # Best-effort: try to open the file using the OS default handler.
    # Suppress stderr/stdout to avoid noisy errors in headless/sandboxed environments.
    try:
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["open", str(html)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform.startswith("win"):
            # type: ignore[attr-defined]
            import os

            os.startfile(str(html))
            proc = None
        else:
            proc = subprocess.run(
                ["xdg-open", str(html)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if proc is not None and proc.returncode == 0:
            print("Opened in your default browser.")
            return
    except Exception:
        pass

    print("If it didn't open automatically, double-click the file above.")


if __name__ == "__main__":
    main()
