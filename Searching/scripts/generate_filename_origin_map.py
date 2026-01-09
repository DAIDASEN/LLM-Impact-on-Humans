import csv
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parent.parent
    arxiv_csv = repo / "papers_arxiv.csv"
    papers_csv = repo / "papers.csv"
    out_csv = repo / "filename_origin_map.csv"

    mapping = {}

    if arxiv_csv.exists():
        with arxiv_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = (r.get("filename") or "").strip()
                if not fn:
                    continue
                mapping[fn] = {
                    "filename": fn,
                    "origin": "arxiv",
                    "arxiv_id": (r.get("arxiv_id") or "").strip(),
                    "pdf_url": (r.get("pdf_url") or "").strip(),
                    "source_csv": "papers_arxiv.csv",
                }

    if papers_csv.exists():
        with papers_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = (r.get("filename") or "").strip()
                if not fn:
                    continue
                if fn in mapping:
                    # already arXiv; note presence in papers.csv
                    mapping[fn]["source_csv"] += ";papers.csv"
                    continue
                src = (r.get("source") or "").strip().lower()
                origin = "arxiv" if src == "arxiv" else "google_scholar" if src else "unknown"
                mapping[fn] = {
                    "filename": fn,
                    "origin": origin,
                    "arxiv_id": "",
                    "pdf_url": "",
                    "source_csv": "papers.csv",
                }

    # write CSV
    rows = list(mapping.values())
    fieldnames = ["filename", "origin", "arxiv_id", "pdf_url", "source_csv"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["filename"]):
            writer.writerow(r)

    # print summary
    total = len(rows)
    n_arxiv = sum(1 for r in rows if r["origin"] == "arxiv")
    n_gs = sum(1 for r in rows if r["origin"] == "google_scholar")
    n_unknown = total - n_arxiv - n_gs
    print(f"Wrote {out_csv}\nTotal filenames: {total}")
    print(f"arXiv: {n_arxiv}, google_scholar: {n_gs}, unknown: {n_unknown}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
