from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _to_bool_or_none(v: Any):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Step1(validate) + Step2(screen_cluster) results.")
    ap.add_argument("--papers_csv", default="papers.csv", help="Original papers.csv (optional but recommended).")
    ap.add_argument("--stage1_jsonl", default="llm_outputs/validate.jsonl")
    ap.add_argument("--stage2_jsonl", default="llm_outputs/screen_cluster.jsonl")
    ap.add_argument("--out_dir", default="llm_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers_csv = Path(args.papers_csv)
    stage1_path = Path(args.stage1_jsonl)
    stage2_path = Path(args.stage2_jsonl)

    base_df = pd.read_csv(papers_csv) if papers_csv.exists() else pd.DataFrame()

    s1_rows = _read_jsonl(stage1_path)
    s2_rows = _read_jsonl(stage2_path)

    # Build Step1 table: filename -> is_paper/is_match + error
    s1_table = []
    for r in s1_rows:
        fn = r.get("filename")
        res = r.get("result") or {}
        s1_table.append(
            {
                "filename": fn,
                "stage1_is_paper": _to_bool_or_none(res.get("is_paper")) if isinstance(res, dict) else None,
                "stage1_is_match": _to_bool_or_none(res.get("is_match")) if isinstance(res, dict) else None,
                "stage1_error": r.get("error"),
            }
        )
    s1_df = pd.DataFrame(s1_table).dropna(subset=["filename"]).drop_duplicates("filename", keep="last")

    # Build Step2 table: filename -> cluster_id/custom_summary + error
    s2_table = []
    for r in s2_rows:
        fn = r.get("filename")
        res = r.get("result") or {}
        s2_table.append(
            {
                "filename": fn,
                "stage2_cluster_id": res.get("cluster_id") if isinstance(res, dict) else None,
                "stage2_custom_summary": res.get("custom_summary") if isinstance(res, dict) else None,
                "stage2_error": r.get("error"),
            }
        )
    s2_df = pd.DataFrame(s2_table).dropna(subset=["filename"]).drop_duplicates("filename", keep="last")

    # Merge into a flat CSV for inspection
    if not base_df.empty and "filename" in base_df.columns:
        merged = base_df.merge(s1_df, on="filename", how="left", suffixes=("", "_s1")).merge(
            s2_df, on="filename", how="left", suffixes=("", "_s2")
        )
    else:
        merged = s1_df.merge(s2_df, on="filename", how="outer")

    # If the input CSV already had stage columns, consolidate to a single canonical column.
    def coalesce(col: str, fallbacks: list[str]) -> None:
        if col in merged.columns:
            return
        for fb in fallbacks:
            if fb in merged.columns:
                merged[col] = merged[fb]
                return

    # Common conflicts when base CSV already has these columns.
    coalesce("stage1_is_paper", ["stage1_is_paper_s1", "stage1_is_paper_x", "stage1_is_paper_y"])
    coalesce("stage1_is_match", ["stage1_is_match_s1", "stage1_is_match_x", "stage1_is_match_y"])
    coalesce("stage1_error", ["stage1_error_s1"])
    coalesce("stage2_cluster_id", ["stage2_cluster_id_s2", "stage2_cluster_id_x", "stage2_cluster_id_y"])
    coalesce("stage2_custom_summary", ["stage2_custom_summary_s2"])
    coalesce("stage2_error", ["stage2_error_s2"])

    merged_out = out_dir / "summary_step1_step2_merged.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")

    # Compute summary numbers
    total = int(merged["filename"].nunique()) if "filename" in merged.columns else 0
    s1_done = int(merged["stage1_is_paper"].notna().sum()) if "stage1_is_paper" in merged.columns else 0
    s1_paper_true = int((merged["stage1_is_paper"] == True).sum()) if "stage1_is_paper" in merged.columns else 0
    s1_match_true = int((merged["stage1_is_match"] == True).sum()) if "stage1_is_match" in merged.columns else 0
    s1_both_true = int(((merged["stage1_is_paper"] == True) & (merged["stage1_is_match"] == True)).sum()) if {"stage1_is_paper","stage1_is_match"}.issubset(merged.columns) else 0
    s1_errors = int(merged["stage1_error"].notna().sum()) if "stage1_error" in merged.columns else 0

    s2_done = int(merged["stage2_cluster_id"].notna().sum()) if "stage2_cluster_id" in merged.columns else 0
    s2_errors = int(merged["stage2_error"].notna().sum()) if "stage2_error" in merged.columns else 0

    # Cluster distribution
    cluster_dist = (
        merged["stage2_cluster_id"]
        .dropna()
        .value_counts(dropna=False)
        .rename_axis("cluster_id")
        .reset_index(name="count")
        if "stage2_cluster_id" in merged.columns
        else pd.DataFrame(columns=["cluster_id", "count"])
    )
    cluster_dist_out = out_dir / "summary_step2_cluster_distribution.csv"
    cluster_dist.to_csv(cluster_dist_out, index=False, encoding="utf-8-sig")

    # Excluded (-1) custom_summary (top reasons)
    excluded_reasons = pd.DataFrame()
    if "stage2_cluster_id" in merged.columns and "stage2_custom_summary" in merged.columns:
        excluded = merged[merged["stage2_cluster_id"] == -1]["stage2_custom_summary"].dropna().astype(str)
        if not excluded.empty:
            excluded_reasons = (
                excluded.value_counts().head(30).rename_axis("reason").reset_index(name="count")
            )
    excluded_out = out_dir / "summary_step2_excluded_reasons_top30.csv"
    excluded_reasons.to_csv(excluded_out, index=False, encoding="utf-8-sig")

    # Write a small JSON report for quick reading
    report = {
        "total_unique_filenames": total,
        "step1": {
            "done": s1_done,
            "is_paper_true": s1_paper_true,
            "is_match_true": s1_match_true,
            "paper_and_match_true": s1_both_true,
            "errors": s1_errors,
        },
        "step2": {
            "done": s2_done,
            "errors": s2_errors,
        },
        "outputs": {
            "merged_csv": str(merged_out),
            "cluster_distribution_csv": str(cluster_dist_out),
            "excluded_reasons_top30_csv": str(excluded_out),
        },
    }
    report_path = out_dir / "summary_step1_step2_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Summary written:")
    print(f"  - {report_path}")
    print(f"  - {merged_out}")
    print(f"  - {cluster_dist_out}")
    print(f"  - {excluded_out}")


if __name__ == "__main__":
    main()


