from __future__ import annotations

# Ensure repository root is on sys.path when running as a module
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse

import pandas as pd
from tqdm import tqdm

from llm_pipeline.openai_client import call_llm
from llm_pipeline.pdf_text import extract_pdf_text
from llm_pipeline.prompts import VALIDATOR_SYSTEM, validator_user_prompt
from llm_pipeline.jsonl_cache import load_processed_filenames, load_result_map_by_filename
from llm_pipeline.utils import SearchRecord, ensure_dir, extract_last_json_object, json_dumps, load_env


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: validate PDF is a paper and matches title/authors.")
    ap.add_argument("--csv", default="papers.csv")
    ap.add_argument("--papers_dir", default="papers")
    ap.add_argument("--out", default="llm_outputs/validate.jsonl")
    ap.add_argument("--resume", action="store_true", help="Skip filenames already present in --out")
    ap.add_argument("--csv_out", default="", help="Write an enriched CSV with stage1 columns")
    ap.add_argument("--inplace", action="store_true", help="If set, overwrite --csv (creates a .bak)")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--head_pages", type=int, default=2)
    ap.add_argument("--tail_pages", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=60000)
    args = ap.parse_args()

    load_env()
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    df = pd.read_csv(args.csv)
    if args.start:
        df = df.iloc[args.start :].reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)

    papers_dir = Path(args.papers_dir)
    processed = load_processed_filenames(out_path) if args.resume else set()
    write_mode = "a" if (args.resume and out_path.exists()) else "w"
    with out_path.open(write_mode, encoding="utf-8") as f:
        for _, r in tqdm(df.iterrows(), total=len(df)):
            rec = SearchRecord.from_pandas_row(r)
            if rec.filename in processed:
                continue
            pdf_path = papers_dir / rec.filename
            if not pdf_path.exists():
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "error": "pdf_not_found",
                    "result": None,
                }
                f.write(json_dumps(row) + "\n")
                continue

            pdf_text = extract_pdf_text(
                pdf_path,
                head_pages=args.head_pages,
                tail_pages=args.tail_pages,
                max_chars=args.max_chars,
            )

            user_prompt = validator_user_prompt(
                pdf_text=pdf_text,
                search_record_json=json_dumps(rec.to_prompt_dict()),
            )

            try:
                raw = call_llm(system_prompt=VALIDATOR_SYSTEM, user_prompt=user_prompt, reasoning_effort="minimal")
                parsed = extract_last_json_object(raw)
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "search_record": rec.to_prompt_dict(),
                    "raw_tail": raw[-2000:],  # for debugging without storing huge content
                    "result": parsed,
                }
            except Exception as e:
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "search_record": rec.to_prompt_dict(),
                    "error": f"{type(e).__name__}: {e}",
                    "result": None,
                }
            f.write(json_dumps(row) + "\n")

    # Optionally write back to CSV (stage1_is_paper / stage1_is_match)
    if args.csv_out or args.inplace:
        result_map = load_result_map_by_filename(out_path)
        df2 = pd.read_csv(args.csv)
        df2["stage1_is_paper"] = df2["filename"].map(lambda x: bool(result_map.get(str(x), {}).get("is_paper")) if str(x) in result_map else None)
        df2["stage1_is_match"] = df2["filename"].map(lambda x: bool(result_map.get(str(x), {}).get("is_match")) if str(x) in result_map else None)

        if args.inplace:
            src = Path(args.csv)
            bak = src.with_suffix(src.suffix + ".bak")
            if not bak.exists():
                src.replace(bak)
                df2.to_csv(src, index=False, encoding="utf-8-sig")
            else:
                df2.to_csv(src, index=False, encoding="utf-8-sig")
        else:
            out_csv = Path(args.csv_out) if args.csv_out else Path("papers_stage1.csv")
            df2.to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
