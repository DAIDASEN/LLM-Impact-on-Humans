from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_pipeline.jsonl_cache import load_result_map_by_filename
from llm_pipeline.utils import ensure_dir, load_env


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _curate_pdfs(
    *,
    df: pd.DataFrame,
    papers_dir: Path,
    curated_dir: Path,
    mode: str,
) -> None:
    """
    mode:
      - "copy": copy PDFs into curated_dir
      - "move": move PDFs into curated_dir (destructive)
    """
    ensure_dir(curated_dir)
    for fn in tqdm(df["filename"].astype(str).tolist(), desc="Curating PDFs"):
        src = papers_dir / fn
        dst = curated_dir / fn
        if not src.exists():
            continue
        if dst.exists():
            continue
        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage1->Stage2->Stage3 with reuse + CSV/PDF operations.")
    ap.add_argument("--csv", default="papers.csv")
    ap.add_argument("--papers_dir", default="papers")
    ap.add_argument("--work_dir", default="llm_outputs")
    ap.add_argument("--curated_dir", default="papers_curated")
    ap.add_argument("--curate_mode", choices=["copy", "move"], default="copy")
    ap.add_argument("--inplace", action="store_true", help="Overwrite papers.csv in-place (creates .bak)")
    args = ap.parse_args()

    load_env()

    csv_path = Path(args.csv)
    papers_dir = Path(args.papers_dir)
    work_dir = ensure_dir(args.work_dir)
    curated_dir = Path(args.curated_dir)

    stage1_jsonl = work_dir / "validate.jsonl"
    stage2_jsonl = work_dir / "screen_cluster.jsonl"
    stage3_jsonl = work_dir / "extract_summary.jsonl"

    # --- Stage 1 ---
    # We call the stage scripts via import-less approach: reuse their CLI by shell is possible,
    # but here we just instruct the user to run them; run_pipeline focuses on data operations.
    # If stage1 output doesn't exist, we fail fast with a helpful message.
    if not stage1_jsonl.exists():
        raise RuntimeError(f"Missing Stage1 output: {stage1_jsonl}. Run: python validate_paper.py --resume --inplace")

    df = pd.read_csv(csv_path)
    stage1_map = load_result_map_by_filename(stage1_jsonl)
    df["stage1_is_paper"] = df["filename"].map(lambda x: bool(stage1_map.get(str(x), {}).get("is_paper")) if str(x) in stage1_map else None)
    df["stage1_is_match"] = df["filename"].map(lambda x: bool(stage1_map.get(str(x), {}).get("is_match")) if str(x) in stage1_map else None)

    curated_df = df[(df["stage1_is_paper"] == True) & (df["stage1_is_match"] == True)].copy()
    curated_csv = work_dir / "papers_curated.csv"
    curated_df.to_csv(curated_csv, index=False, encoding="utf-8-sig")

    _curate_pdfs(df=curated_df, papers_dir=papers_dir, curated_dir=curated_dir, mode=args.curate_mode)

    # --- Stage 2 ---
    if stage2_jsonl.exists():
        stage2_map = load_result_map_by_filename(stage2_jsonl)
        df["stage2_cluster_id"] = df["filename"].map(lambda x: stage2_map.get(str(x), {}).get("cluster_id") if str(x) in stage2_map else None)
        df["stage2_custom_summary"] = df["filename"].map(lambda x: stage2_map.get(str(x), {}).get("custom_summary") if str(x) in stage2_map else None)

    # --- Stage 3 ---
    if stage3_jsonl.exists():
        stage3_map = load_result_map_by_filename(stage3_jsonl)
        df["stage3_result_json"] = df["filename"].map(lambda x: _json_dumps(stage3_map.get(str(x))) if str(x) in stage3_map else None)

    # Write back
    if args.inplace:
        bak = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not bak.exists():
            csv_path.replace(bak)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        out_csv = work_dir / "papers_enriched.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()


