from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    # Try utf-8-sig first; fall back to default if needed.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _norm_str(s: object) -> str:
    if s is None:
        return ""
    v = str(s).strip()
    if v.lower() == "nan":
        return ""
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two step3 export CSVs and summarize differences.")
    ap.add_argument("--a", default="llm_outputs/step3_export_no_excluded.csv")
    ap.add_argument("--b", default="llm_outputs/step3_export.csv")
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)

    if not a_path.exists():
        raise FileNotFoundError(a_path)
    if not b_path.exists():
        raise FileNotFoundError(b_path)

    a = _read_csv(a_path)
    b = _read_csv(b_path)

    a_cols = list(a.columns)
    b_cols = list(b.columns)

    only_a_cols = [c for c in a_cols if c not in b_cols]
    only_b_cols = [c for c in b_cols if c not in a_cols]

    print("=== Schema ===")
    print(f"A: {a_path.as_posix()} rows={len(a)} cols={len(a_cols)}")
    print(f"B: {b_path.as_posix()} rows={len(b)} cols={len(b_cols)}")
    if only_a_cols:
        print(f"Columns only in A: {only_a_cols}")
    if only_b_cols:
        print(f"Columns only in B: {only_b_cols}")

    # Content comparison by stable keys
    key_candidates = []
    if "URL" in a.columns and "URL" in b.columns:
        key_candidates.append("URL")
    if "Title" in a.columns and "Title" in b.columns:
        key_candidates.append("Title")

    print("\n=== Content keys ===")
    if not key_candidates:
        print("No shared key columns (URL/Title) found; cannot compare content reliably.")
        return

    # Build keys set. Prefer URL; fall back to Title if URL missing.
    def build_keys(df: pd.DataFrame) -> set[str]:
        keys: set[str] = set()
        if "URL" in df.columns:
            for v in df["URL"].tolist():
                s = _norm_str(v)
                if s:
                    keys.add("URL::" + s)
        if not keys and "Title" in df.columns:
            for v in df["Title"].tolist():
                s = _norm_str(v)
                if s:
                    keys.add("TITLE::" + s)
        return keys

    a_keys = build_keys(a)
    b_keys = build_keys(b)

    only_in_a = sorted(a_keys - b_keys)
    only_in_b = sorted(b_keys - a_keys)

    print(f"A unique keys: {len(a_keys)}")
    print(f"B unique keys: {len(b_keys)}")
    print(f"Only in A: {len(only_in_a)}")
    print(f"Only in B: {len(only_in_b)}")

    # Small preview
    def preview(lst: list[str], n: int = 5) -> list[str]:
        return lst[:n]

    if only_in_a:
        print("Only-in-A sample:", preview(only_in_a))
    if only_in_b:
        print("Only-in-B sample:", preview(only_in_b))

    # Missingness snapshot for key columns
    print("\n=== Missingness (selected columns) ===")
    for col in ["filename", "URL", "Title"]:
        if col in a.columns:
            miss = a[col].isna().sum() + (a[col].astype(str).str.strip().str.lower() == "nan").sum()
            print(f"A {col}: missing~{int(miss)}/{len(a)}")
        if col in b.columns:
            miss = b[col].isna().sum() + (b[col].astype(str).str.strip().str.lower() == "nan").sum()
            print(f"B {col}: missing~{int(miss)}/{len(b)}")


if __name__ == "__main__":
    main()
