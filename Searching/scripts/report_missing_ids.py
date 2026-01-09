from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_ID_RE = re.compile(r"^(\d{4})\.pdf$", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report missing 4-digit PDF IDs in a directory.")
    ap.add_argument("--dir", default="data/papers/google_scholar", help="Directory containing PDFs like 0000.pdf")
    ap.add_argument("--start", type=int, default=0, help="Start ID (inclusive)")
    ap.add_argument("--end", type=int, default=399, help="End ID (inclusive)")
    ap.add_argument("--out", default="data/csv/google_scholar/missing_0000_0399.csv")
    args = ap.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    existing: set[int] = set()
    for p in base_dir.iterdir():
        if not p.is_file():
            continue
        m = _ID_RE.match(p.name)
        if not m:
            continue
        existing.add(int(m.group(1)))

    missing_rows = []
    for i in range(int(args.start), int(args.end) + 1):
        if i not in existing:
            missing_rows.append({"id": f"{i:04d}", "expected_filename": f"{i:04d}.pdf"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Scanned: {base_dir}")
    print(f"Existing: {len(existing)}")
    print(f"Missing:  {len(missing_rows)}")
    print(f"Wrote:    {out_path}")


if __name__ == "__main__":
    main()
