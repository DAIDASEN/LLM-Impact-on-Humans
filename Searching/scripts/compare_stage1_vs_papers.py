import argparse
from pathlib import Path

import pandas as pd


def _norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find rows that appear in stage1_titles.csv but not in papers.csv, and export to a CSV."
        )
    )
    parser.add_argument(
        "--stage1",
        default=str(Path("data/csv/google_scholar/stage1_titles.csv")),
        help="Path to stage1_titles.csv",
    )
    parser.add_argument(
        "--papers",
        default=str(Path("data/csv/google_scholar/papers.csv")),
        help="Path to papers.csv",
    )
    parser.add_argument(
        "--out",
        default=str(Path("data/csv/google_scholar/failed.csv")),
        help="Output CSV path (default: data/csv/google_scholar/failed.csv)",
    )
    parser.add_argument(
        "--only-title-url",
        action="store_true",
        help="If set, only export two columns: name and url.",
    )
    args = parser.parse_args()

    stage1_path = Path(args.stage1)
    papers_path = Path(args.papers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_df = pd.read_csv(stage1_path)
    papers_df = pd.read_csv(papers_path)

    # Prefer title_norm; fall back to title.
    stage1_key_col = "title_norm" if "title_norm" in stage1_df.columns else "title"
    papers_key_col = "title_norm" if "title_norm" in papers_df.columns else "title"

    stage1_keys = stage1_df[stage1_key_col].map(_norm_text)
    papers_key_set = set(papers_df[papers_key_col].map(_norm_text).tolist())

    missing_mask = ~stage1_keys.isin(papers_key_set)
    failed_df = stage1_df.loc[missing_mask].copy()

    # Helpful summary columns (kept minimal).
    failed_df.insert(0, "match_key", stage1_keys[missing_mask].values)

    if args.only_title_url:
        title_col = "title" if "title" in failed_df.columns else stage1_key_col
        url_col = "url_gs" if "url_gs" in failed_df.columns else ("url" if "url" in failed_df.columns else "")
        if not url_col:
            slim_df = pd.DataFrame({"name": failed_df[title_col].astype(str)})
        else:
            slim_df = pd.DataFrame(
                {
                    "name": failed_df[title_col].astype(str),
                    "url": failed_df[url_col].astype(str),
                }
            )
        slim_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        failed_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Stage1 rows: {len(stage1_df)}")
    print(f"Papers rows: {len(papers_df)}")
    print(f"Missing in papers: {len(failed_df)}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
