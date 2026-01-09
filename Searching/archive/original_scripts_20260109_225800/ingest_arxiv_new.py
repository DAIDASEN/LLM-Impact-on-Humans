from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


def normalize_title(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[\[\]\(\)\{\}\.,:;!?\"'“”‘’`´\-–—_/\\|<>~^+=*@#$%&]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _authors_to_list_str(authors: str) -> str:
    parts = [a.strip() for a in (authors or "").split(";") if a.strip()]
    return str(parts)


def download_pdf(url: str, out_path: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "arXivDownloader/1.0"})
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct.lower():
            return False, f"non_pdf_content_type:{ct}"
        out_path.write_bytes(resp.content)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest arXiv CSV into a local papers_arxiv/ folder and papers_arxiv.csv (new-only).")
    ap.add_argument("--arxiv_csv", default="search_ieee_pipeline/sources/arxiv/papers_for_pipeline_arxiv.csv")
    ap.add_argument("--out_csv", default="papers_arxiv.csv")
    ap.add_argument("--papers_dir", default="papers_arxiv")
    ap.add_argument("--download", action="store_true", help="Download missing PDFs into papers_dir.")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    arxiv_csv = Path(args.arxiv_csv)
    if not arxiv_csv.exists():
        raise FileNotFoundError(f"Missing arXiv CSV: {arxiv_csv}")

    papers_dir = Path(args.papers_dir)
    out_csv = Path(args.out_csv)

    # Existing arxiv table (for de-dup)
    existing_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    existing_ids = set()
    if not existing_df.empty and "arxiv_id" in existing_df.columns:
        existing_ids = set(existing_df["arxiv_id"].astype(str).tolist())

    # Also de-dup against already downloaded PDFs
    existing_pdf_ids = set()
    if papers_dir.exists():
        for p in papers_dir.glob("*.pdf"):
            existing_pdf_ids.add(p.stem)

    src_df = pd.read_csv(arxiv_csv)
    if args.limit and args.limit > 0:
        src_df = src_df.head(args.limit).copy()

    # Load google_scholar titles/urls to avoid duplicates (optional best-effort)
    gs_title_norm = set()
    gs_arxiv_ids = set()
    papers_csv = Path("papers.csv")
    if papers_csv.exists():
        gs_df = pd.read_csv(papers_csv)
        if "title_norm" in gs_df.columns:
            gs_title_norm = set(gs_df["title_norm"].astype(str).tolist())
        elif "title" in gs_df.columns:
            gs_title_norm = set(gs_df["title"].astype(str).map(normalize_title).tolist())
        if "url_gs" in gs_df.columns:
            for u in gs_df["url_gs"].astype(str).tolist():
                m = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", u)
                if m:
                    gs_arxiv_ids.add(m.group(1))

    new_rows: list[dict[str, Any]] = []
    download_report_rows: list[dict[str, Any]] = []

    for _, r in tqdm(src_df.iterrows(), total=len(src_df), desc="Selecting new arXiv records"):
        arxiv_id = str(r.get("arxiv_id", "")).strip()
        title = str(r.get("title", "")).strip()
        pdf_url = str(r.get("pdf_url", "")).strip()
        published = str(r.get("published", "")).strip()
        authors = str(r.get("authors", "")).strip()

        if not arxiv_id or not pdf_url or not title:
            continue

        # De-dup
        if arxiv_id in existing_ids or arxiv_id in existing_pdf_ids or arxiv_id in gs_arxiv_ids:
            continue
        if normalize_title(title) in gs_title_norm:
            continue

        # Decide filename scheme: arxiv_id.pdf (no collisions, source-separated folder)
        filename = f"{arxiv_id}.pdf"
        pdf_path = papers_dir / filename

        ok = True
        err = ""
        if args.download and not pdf_path.exists():
            ok, err = download_pdf(pdf_url, pdf_path)
            download_report_rows.append(
                {"arxiv_id": arxiv_id, "pdf_url": pdf_url, "filename": filename, "ok": ok, "error": err}
            )
            if not ok:
                # Skip records we couldn't download (otherwise downstream fails)
                continue

        # Parse year from published (ISO timestamp)
        year = ""
        if published:
            try:
                year = str(datetime.fromisoformat(published.replace("Z", "+00:00")).year)
            except Exception:
                year = published[:4]

        new_rows.append(
            {
                "search_cluster": "",
                "search_cluster_name": "arxiv",
                "title": title,
                "title_norm": normalize_title(title),
                "year_gs": year,
                "venue_gs": "arXiv",
                "author_gs": _authors_to_list_str(authors),
                "url_gs": pdf_url.replace("/pdf/", "/abs/").replace(".pdf", ""),
                "source": "arxiv",
                "row_id": None,
                "filename": filename,
                "stage1_is_paper": None,
                "stage1_is_match": None,
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
                "published": published,
            }
        )

    if not new_rows:
        print("No new arXiv records found (nothing to ingest).")
        return

    # Assign row_id within arxiv table
    start_id = 0
    if not existing_df.empty and "row_id" in existing_df.columns:
        try:
            start_id = int(pd.to_numeric(existing_df["row_id"], errors="coerce").max()) + 1
        except Exception:
            start_id = 0
    for i, rr in enumerate(new_rows):
        rr["row_id"] = start_id + i

    out_df_new = pd.DataFrame(new_rows)
    out_df = pd.concat([existing_df, out_df_new], ignore_index=True) if not existing_df.empty else out_df_new
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote/updated: {out_csv} (+{len(out_df_new)} new rows)")

    if download_report_rows:
        report_path = Path("llm_outputs") / "arxiv_download_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["arxiv_id", "pdf_url", "filename", "ok", "error"])
            if f.tell() == 0:
                w.writeheader()
            for rr in download_report_rows:
                w.writerow(rr)
        print(f"Download report appended: {report_path}")


if __name__ == "__main__":
    main()




