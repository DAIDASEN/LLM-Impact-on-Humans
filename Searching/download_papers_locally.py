#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import pandas as pd


def read_csv_auto(path: str) -> pd.DataFrame:
    # 兼容 Windows/Excel 常见编码；兜底不崩溃
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")  # pandas 支持 encoding_errors :contentReference[oaicite:2]{index=2}


def collect_ids_from_pdfs(papers_dir: Path) -> set[int]:
    # 支持：0034.pdf / 0034_xxx.pdf / 0034-xxx.pdf（只要开头是数字即可）
    pat = re.compile(r"^(\d+)")
    ids: set[int] = set()
    for p in papers_dir.glob("*.pdf"):  # pathlib.glob :contentReference[oaicite:3]{index=3}
        m = pat.match(p.stem)
        if m:
            ids.add(int(m.group(1)))
    return ids


def main():
    ap = argparse.ArgumentParser(description="Export rows from stage1.csv that have corresponding PDFs in papers/")
    ap.add_argument("--stage1", default="stage1.csv", help="Path to stage1 CSV (default: stage1.csv)")
    ap.add_argument("--papers", default="papers", help="Papers folder (default: papers)")
    ap.add_argument("--out", default="paper_list_downloaded.csv", help="Output CSV (default: paper_list_downloaded.csv)")
    ap.add_argument("--digits", type=int, default=4, help="Zero pad digits for filename column (default: 4 => 0034.pdf)")
    args = ap.parse_args()

    stage1_path = Path(args.stage1)
    papers_dir = Path(args.papers)
    out_path = Path(args.out)

    if not stage1_path.exists():
        raise SystemExit(f"stage1 CSV not found: {stage1_path.resolve()}")
    if not papers_dir.exists():
        raise SystemExit(f"papers folder not found: {papers_dir.resolve()}")

    df = read_csv_auto(str(stage1_path))
    df = df.copy()

    # 原始行号作为 paper id（与你的 0000.pdf 命名规则一致）
    df["row_id"] = df.index

    downloaded_ids = collect_ids_from_pdfs(papers_dir)

    # 生成对应文件名列（方便回溯）
    df["filename"] = df["row_id"].apply(lambda i: f"{i:0{args.digits}d}.pdf")

    # 只保留已下载的行（pandas isin 用 set 很快） :contentReference[oaicite:4]{index=4}
    downloaded_df = df[df["row_id"].isin(downloaded_ids)].copy()

    # 可选：按 row_id 排序，保持和文件编号一致
    downloaded_df = downloaded_df.sort_values("row_id")

    downloaded_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Detected downloaded PDFs: {len(downloaded_ids)}")
    print(f"Exported rows: {len(downloaded_df)}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
