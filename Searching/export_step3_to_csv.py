from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from llm_pipeline.pdf_text import extract_pdf_text


CLUSTER_NAME = {
    0: "Social & Collaboration",
    1: "Psychology & Persuasion",
    2: "Creativity & Ideation",
    3: "Education & Productivity",
    4: "None of the above",
    -1: "Excluded",
}


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


def _join_list(v: Any, sep: str = "; ") -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return sep.join(str(x).strip() for x in v if str(x).strip())
    return str(v)


def _format_human_n(v: Any) -> str:
    if not isinstance(v, dict):
        return "" if v in (None, "NA") else str(v)
    value = v.get("value", None)
    unit = v.get("unit", None)
    notes = v.get("notes", None)
    parts = []
    if value is not None:
        parts.append(str(value))
    if unit and unit != "NA":
        parts.append(str(unit))
    base = " ".join(parts).strip()
    if notes and str(notes).strip() and str(notes).strip() != "NA":
        return f"{base} ({str(notes).strip()})" if base else str(notes).strip()
    return base


_ABSTRACT_RE = re.compile(
    r"(?is)\babstract\b\s*[:\n\r]+([\s\S]{50,3000}?)(?:\n\s*(?:keywords|index terms|introduction)\b|$)"
)


def _extract_abstract_from_pdf(pdf_path: Path) -> str:
    """
    Local heuristic: extract 'Abstract' section from the first pages.
    No API call. Returns "" if not found.
    """
    try:
        txt = extract_pdf_text(pdf_path, head_pages=2, tail_pages=0, max_chars=30000)
    except Exception:
        return ""
    m = _ABSTRACT_RE.search(txt)
    if not m:
        return ""
    abs_text = re.sub(r"\s+", " ", m.group(1)).strip()
    # Keep it reasonably short for CSV readability
    return abs_text[:2000]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Step3 jsonl into a flat CSV with specified columns.")
    ap.add_argument("--papers_csv", default="papers.csv", help="Source CSV (for Publication/Year/URL/Title fallback).")
    ap.add_argument("--step3_jsonl", default="llm_outputs/extract_summary.jsonl")
    ap.add_argument("--out_csv", default="llm_outputs/step3_export.csv")
    ap.add_argument("--include_abstract", action="store_true", help="Try to extract Abstract from local PDFs (no API).")
    ap.add_argument("--papers_dir", default="papers", help="PDF directory used when --include_abstract is set.")
    ap.add_argument(
        "--drop_excluded",
        action="store_true",
        default=True,
        help="Drop rows whose Step2 cluster_id is -1 (Excluded). Default: True.",
    )
    args = ap.parse_args()

    papers_df = pd.read_csv(args.papers_csv)
    if "filename" not in papers_df.columns:
        raise ValueError("papers_csv must include a 'filename' column.")
    papers_df["filename"] = papers_df["filename"].astype(str)
    # Some sources (e.g., arXiv refresh) may contain duplicate filenames; keep the last.
    papers_df = papers_df.drop_duplicates(subset=["filename"], keep="last").copy()
    papers_map = papers_df.set_index("filename").to_dict(orient="index")

    step3_path = Path(args.step3_jsonl)
    rows = _read_jsonl(step3_path)

    out_rows: list[dict[str, Any]] = []
    papers_dir = Path(args.papers_dir)

    for r in tqdm(rows, desc="Exporting"):
        fn = str(r.get("filename", "")).strip()
        if not fn:
            continue

        base = papers_map.get(fn, {})
        search_record = r.get("search_record") or base

        cluster_assignment = r.get("cluster_assignment") or {}
        cluster_id = cluster_assignment.get("cluster_id", None)
        if cluster_id is None:
            # fallback: if Step2 was written back to papers.csv
            cluster_id = base.get("stage2_cluster_id", None)

        if args.drop_excluded:
            try:
                if int(cluster_id) == -1:
                    continue
            except Exception:
                pass

        result = r.get("result")
        error = r.get("error")
        skipped = r.get("skipped")
        reason = r.get("reason")

        # If Step3 didn't run or failed, still export a row with Remark.
        title = ""
        keywords = ""
        llm = ""
        human_n = ""
        interaction = ""
        long_term = ""
        conclusion = ""
        if isinstance(result, dict):
            title = str(result.get("Title", "") or "")
            keywords = _join_list(result.get("Keywords"), sep="; ")
            llm = _join_list(result.get("LLM"), sep="; ")
            human_n = _format_human_n(result.get("Human_N"))
            interaction = str(result.get("Interaction", "") or "")
            long_term = str(result.get("Long_term", "") or "")
            conclusion = _join_list(result.get("Conclusion"), sep=" | ")

        # Fill fallback title from csv if needed
        if not title:
            title = str(search_record.get("title") or base.get("title") or "")

        publication = str(search_record.get("venue_gs") or base.get("venue_gs") or "")
        year = search_record.get("year_gs", base.get("year_gs", ""))
        url = str(search_record.get("url_gs") or base.get("url_gs") or "")

        class_label = ""
        if cluster_id is not None:
            try:
                cid_int = int(cluster_id)
                class_label = f"{cid_int} - {CLUSTER_NAME.get(cid_int, '')}".strip(" -")
            except Exception:
                class_label = str(cluster_id)

        remark_parts = []
        if isinstance(cluster_assignment, dict) and cluster_assignment.get("custom_summary"):
            remark_parts.append(f"stage2: {cluster_assignment.get('custom_summary')}")
        if skipped:
            remark_parts.append(f"skipped: {reason}")
        if error:
            remark_parts.append(f"error: {error}")
        remark = " | ".join(str(x) for x in remark_parts if str(x).strip())

        abstract = ""
        if args.include_abstract:
            pdf_path = papers_dir / fn
            if pdf_path.exists():
                abstract = _extract_abstract_from_pdf(pdf_path)

        out_rows.append(
            {
                "Publication": publication,
                "Year": year,
                "URL": url,
                "Title": title,
                "Class": class_label,
                "Keywords": keywords,
                "LLM": llm,
                "Human N": human_n,
                "Interaction": interaction,
                "Long-term?": long_term,
                "Conclusion": conclusion,
                "Remark": remark,
                "Abstract": abstract,
                # keep filename for traceability
                "filename": fn,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


