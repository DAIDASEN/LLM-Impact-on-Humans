from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_origin_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError(f"Missing column 'filename' in {path}")
    if "origin" in df.columns:
        origin_col = "origin"
    elif "source" in df.columns:
        origin_col = "source"
    else:
        raise ValueError(f"Missing column 'origin' (or 'source') in {path}")

    m: dict[str, str] = {}
    for _, r in df.iterrows():
        fn = str(r["filename"]).strip()
        if not fn or fn.lower() == "nan":
            continue
        m[fn] = str(r[origin_col]).strip()
    return m


def _count_for_csv(csv_path: Path, origin_map: dict[str, str]) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        return {"total_rows": len(df), "has_filename": 0, "arxiv": 0, "google_scholar": 0, "unknown": 0}

    fns = df["filename"].astype(str).map(str.strip)
    total = len(fns)
    arxiv = 0
    gs = 0
    unknown = 0
    for fn in fns:
        if not fn or fn.lower() == "nan":
            unknown += 1
            continue
        origin = origin_map.get(fn, "unknown")
        if origin == "arxiv":
            arxiv += 1
        elif origin in ("google_scholar", "gs", "google"):
            gs += 1
        else:
            # treat anything else as unknown (keeps it conservative)
            unknown += 1

    return {
        "total_rows": total,
        "has_filename": total,
        "arxiv": arxiv,
        "google_scholar": gs,
        "unknown": unknown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Count arXiv vs Google Scholar items in pipeline outputs.")
    ap.add_argument(
        "--origin_map",
        default="filename_origin_map.csv",
        help="CSV containing filename->origin mapping (expects columns: filename + origin).",
    )
    ap.add_argument(
        "--outputs",
        nargs="*",
        default=[
            "llm_outputs/step3_export_no_excluded.csv",
            "llm_outputs/step3_export.csv",
            "llm_outputs/papers_curated.csv",
            "llm_outputs/papers_enriched.csv",
        ],
        help="One or more pipeline output CSVs to count.",
    )
    args = ap.parse_args()

    origin_map_path = Path(args.origin_map)
    if not origin_map_path.exists():
        raise FileNotFoundError(f"Missing origin map: {origin_map_path}")
    origin_map = _read_origin_map(origin_map_path)

    rows: list[dict[str, object]] = []
    for p in args.outputs:
        path = Path(p)
        if not path.exists():
            continue
        c = _count_for_csv(path, origin_map)
        rows.append({"file": str(path).replace("\\", "/"), **c})

    if not rows:
        print("No output files found.")
        return

    out_df = pd.DataFrame(rows)
    # Stable column order
    cols = ["file", "total_rows", "arxiv", "google_scholar", "unknown"]
    out_df = out_df[cols]
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
