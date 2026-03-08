from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


def append_row(path: Path, row: Mapping[str, object], *, preferred_header: list[str] | None = None) -> None:
    """
    Append one dict row to a CSV file, automatically upgrading the header when new
    columns appear.

    This avoids corrupting existing CSVs when we add new metrics over time.
    """

    row_dict = {str(k): ("" if v is None else v) for k, v in row.items()}

    if not path.exists():
        header = list(row_dict.keys())
        if preferred_header:
            all_keys = set(header)
            ordered = [k for k in preferred_header if k in all_keys]
            ordered_set = set(ordered)
            ordered += [k for k in header if k not in ordered_set]
            header = ordered
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerow({k: row_dict.get(k, "") for k in header})
        return

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_header = list(reader.fieldnames or [])
        existing_rows = [dict(r) for r in reader]

    if not existing_header:
        header = list(row_dict.keys())
        if preferred_header:
            all_keys = set(header)
            header = [k for k in preferred_header if k in all_keys] + sorted(all_keys - set(preferred_header))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerow({k: row_dict.get(k, "") for k in header})
        return

    missing_keys = [k for k in row_dict.keys() if k not in existing_header]
    if not missing_keys:
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=existing_header)
            w.writerow({k: row_dict.get(k, "") for k in existing_header})
        return

    # Upgrade schema: rewrite file with a unified header.
    all_keys = set(existing_header) | set(row_dict.keys())
    if preferred_header:
        header: list[str] = [k for k in preferred_header if k in all_keys]
        header_set = set(header)
        # Preserve existing column order for the remaining known keys.
        for k in existing_header:
            if k in all_keys and k not in header_set:
                header.append(k)
                header_set.add(k)
        header += sorted(all_keys - header_set)
    else:
        header = existing_header + sorted(all_keys - set(existing_header))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in existing_rows:
            w.writerow({k: r.get(k, "") for k in header})
        w.writerow({k: row_dict.get(k, "") for k in header})

