from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_pipeline.openai_client import call_llm
from llm_pipeline.pdf_text import extract_pdf_text
from llm_pipeline.prompts import EXTRACT_SYSTEM, extract_user_prompt
from llm_pipeline.jsonl_cache import load_processed_filenames, load_result_map_by_filename
from llm_pipeline.utils import SearchRecord, ensure_dir, extract_last_json_object, json_dumps, load_env


def _load_cluster_map(screen_jsonl: Path) -> dict[str, dict]:
    """
    Map filename -> {cluster_id, custom_summary, ...}
    """
    m: dict[str, dict] = {}
    with screen_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fn = obj.get("filename")
            res = obj.get("result")
            if fn and isinstance(res, dict):
                m[fn] = res
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: extract structured human-impact summary based on cluster assignment.")
    ap.add_argument("--csv", default="papers.csv")
    ap.add_argument("--papers_dir", default="papers")
    ap.add_argument("--screen_jsonl", default="llm_outputs/screen_cluster.jsonl")
    ap.add_argument("--out", default="llm_outputs/extract_summary.jsonl")
    ap.add_argument("--resume", action="store_true", help="Skip filenames already present in --out")
    ap.add_argument("--csv_out", default="", help="Write an enriched CSV with stage3 (nested JSON) column")
    ap.add_argument("--inplace", action="store_true", help="If set, overwrite --csv (creates a .bak)")
    ap.add_argument("--only_included", action="store_true", help="Only run for cluster_id in {0,1,2,3,4}")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--head_pages", type=int, default=4)
    ap.add_argument("--tail_pages", type=int, default=4)
    ap.add_argument("--max_chars", type=int, default=80000)
    args = ap.parse_args()

    load_env()
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    cluster_map = _load_cluster_map(Path(args.screen_jsonl))

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

            assign = cluster_map.get(rec.filename)
            if not assign:
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "error": "missing_cluster_assignment",
                    "result": None,
                }
                f.write(json_dumps(row) + "\n")
                continue

            cluster_id = assign.get("cluster_id", None)
            if args.only_included and cluster_id == -1:
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "skipped": True,
                    "reason": "excluded_cluster_id_-1",
                    "cluster_assignment": assign,
                    "result": None,
                }
                f.write(json_dumps(row) + "\n")
                continue

            pdf_path = papers_dir / rec.filename
            if not pdf_path.exists():
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "error": "pdf_not_found",
                    "cluster_assignment": assign,
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

            user_prompt = extract_user_prompt(
                pdf_text=pdf_text,
                cluster_assignment_json=json_dumps(assign),
            )

            try:
                raw = call_llm(system_prompt=EXTRACT_SYSTEM, user_prompt=user_prompt, reasoning_effort="minimal")
                parsed = extract_last_json_object(raw)
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "search_record": rec.to_prompt_dict(),
                    "cluster_assignment": assign,
                    "raw_tail": raw[-2000:],
                    "result": parsed,
                }
            except Exception as e:
                row = {
                    "row_id": rec.row_id,
                    "filename": rec.filename,
                    "search_record": rec.to_prompt_dict(),
                    "cluster_assignment": assign,
                    "error": f"{type(e).__name__}: {e}",
                    "result": None,
                }
            f.write(json_dumps(row) + "\n")

    # Optionally write back to CSV (stage3_result_json)
    if args.csv_out or args.inplace:
        result_map = load_result_map_by_filename(out_path)
        df2 = pd.read_csv(args.csv)
        # Store the stage3 structured JSON as a compact JSON string to keep CSV flat
        df2["stage3_result_json"] = df2["filename"].map(lambda x: json.dumps(result_map.get(str(x)), ensure_ascii=False) if str(x) in result_map else None)

        if args.inplace:
            src = Path(args.csv)
            bak = src.with_suffix(src.suffix + ".bak")
            if not bak.exists():
                src.replace(bak)
                df2.to_csv(src, index=False, encoding="utf-8-sig")
            else:
                df2.to_csv(src, index=False, encoding="utf-8-sig")
        else:
            out_csv = Path(args.csv_out) if args.csv_out else Path("papers_stage3.csv")
            df2.to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()


